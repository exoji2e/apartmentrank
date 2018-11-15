import logging
import pathlib
import pickle
import re
from typing import Dict, Any, Tuple, Optional, List
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass
from math import atan2, degrees

from colorama import Fore, Style
from geopy import distance
from geopy.geocoders import Nominatim
from bs4 import BeautifulSoup
from tabulate import tabulate

import requests
import feedparser

# from joblib import Memory
# memory = Memory('.joblibcache', verbose=0)

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("geopy").setLevel(logging.INFO)
logging.getLogger("urllib3").setLevel(logging.INFO)
log = logging.getLogger(__name__)

CITY = "Lund"

malmo_c_coords = (55.6091, 12.9999)
triangeln_coords = (55.5944, 13.0004)
lund_c_coords = (55.7068, 13.187)
lth_coords = (55.7124, 13.2091)


DOWNPAYMENT = 1_050_000
LOAN_FACTOR = None  # 0.7
INTEREST = 0.015
OPPORTUNITY_COC_RATE = 0.00


class Datastore:
    path = pathlib.Path('./datastore.pickle')
    data: Dict[str, Dict[str, Any]] = defaultdict(dict)

    def __init__(self) -> None:
        if self.path.exists():
            with self.path.open("rb") as f:
                data = pickle.load(f)
                self.data.update(data)

    def __enter__(self):
        pass

    def __exit__(self, *args, **kwargs):
        self.save()

    def save(self) -> None:
        with open(self.path, "wb+") as f:
            pickle.dump(dict(self.data), f)


def degrees_to_direction(deg: float):
    dirs = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
    i = round((((deg + 360) % 360) / 360) * len(dirs))
    return dirs[i % (len(dirs))]


def test_degrees_to_direction():
    assert all(degrees_to_direction(d) == "E" for d in (-20, 0, 20))
    assert all(degrees_to_direction(d) == "NE" for d in [30, 45])
    assert all(degrees_to_direction(d) == "N" for d in [90])
    assert all(degrees_to_direction(d) == "NW" for d in [90 + 45])
    assert all(degrees_to_direction(d) == "W" for d in [180])
    assert all(degrees_to_direction(d) == "S" for d in [270])


cities: Dict[str, Dict[str, Any]] = {
    "Lund": {
        "center": lund_c_coords,
        "attractors": [lund_c_coords],
        # attractors = [center, lth_coords]
    },
    "Malmö": {
        "center": malmo_c_coords,
        "attractors": [malmo_c_coords, triangeln_coords],
    }
}


@dataclass
class Property:
    link: str
    address: str
    price: float
    area: float
    rooms: float
    monthly_fee: float
    published: Optional[datetime] = None
    coords: Optional[Tuple[float, float]] = None
    city: Optional[str] = None

    def __repr__(self):
        return f"<{self.address},\n  Area: {self.area}\n  Rooms: {self.rooms}\n  MoCost: {self.monthly_cost()}\n  MoCost/m²: {self.monthly_cost_per_sqm()}>"

    @staticmethod
    def headers() -> List[str]:
        return ["address", "price", "area", "fee", "rooms", "cst/mo", "cst/m²/mo", "dist", "dir", "publ"]

    def row(self) -> List[Any]:
        attractors = cities[CITY]["attractors"]
        dist = round(sum(self.distance_to(loc) or 0 for loc in attractors), 1) or None
        direction = self.direction(lund_c_coords)
        return [
            self.address,
            #self.link,
            self.price,
            self.area,
            self.monthly_fee,
            "x" * int(self.rooms) + "." * (round(self.rooms) - int(self.rooms)),  # + f" ({self.rooms})",
            round(self.monthly_cost()),
            round(self.monthly_cost_per_sqm(), 1),
            dist,
            direction,
            f"{str(self.time_since_published().days) + 'd ago' if self.time_since_published() else ''}",
        ]

    def monthly_cost_per_sqm(self) -> float:
        return self.monthly_cost() / self.area

    def monthly_cost(self, downpayment=DOWNPAYMENT, interest=INTEREST, taxreduction=0.3, loan_factor=LOAN_FACTOR) -> float:
        loaned = loan_factor * self.price if loan_factor else max(0, self.price - downpayment)
        mortgage = loaned * (interest * (1 - taxreduction) / 12)
        cost_of_capital = (self.price - loaned) * OPPORTUNITY_COC_RATE / 12
        return mortgage + self.monthly_fee + cost_of_capital

    def direction(self, other: Tuple[float, float]) -> Optional[str]:
        if self.coords:
            d = atan2(self.coords[0] - other[0], self.coords[1] - other[1])
            return degrees_to_direction(degrees(d))
        return None

    def distance_to(self, other: Tuple[float, float]) -> Optional[float]:
        if not self.coords:
            return None
        return distance.distance(self.coords, other).km

    def time_since_published(self) -> Optional[timedelta]:
        return datetime.now() - self.published if self.published else None


def get_entry(link: str) -> str:
    if "page" not in db.data[link]:
        r = requests.get(link)
        with db:
            db.data[link].update(page=r.content)
    else:
        log.debug(f"Got {link} from cache")
    return db.data[link]["page"]


def parse_page(title, link):
    txt = get_entry(link)
    # print(txt)
    soup = BeautifulSoup(txt, 'html.parser')
    # print(soup)

    priceprop = soup.find(class_="property__price")
    if priceprop:
        price = float("".join(priceprop.string.strip().split(" ")[:-1]))
        monthly_fee = 0
        for p in soup.find(class_="property__attributes-container").find_all("dd"):
            txt = p.string
            if not txt:
                txt = p.contents[0]
            # &nbsp; to space
            txt = txt.replace("\xa0", " ")
            if txt.split()[-1] == "m²":
                sqm = float(txt.split(" ")[0].replace(",", "."))
            elif "rum" in txt:
                rooms = float(txt.split(" ")[0].replace(",", "."))
            elif "kr/m" in txt and "n" in txt:
                num = "".join(txt.strip().split(" ")[:-1])
                monthly_fee = float(num.replace(",", "."))

        return Property(link, title, price, sqm, rooms, monthly_fee)
    else:
        cprint("Probably already sold", Fore.YELLOW)
        return


def get_coord(address, city=None) -> Optional[Tuple[float, float]]:
    geolocator = Nominatim(user_agent="apartmentbuyer", timeout=10)
    try:
        address = address.split("lgh")[0].split(",")[0] + f", {city or 'Skane'}"

        # Remove spaces in building/apartment ID 'Streetname 2 A' to just 'Streetname 2A'
        address = re.sub('([0-9]+) ([A-Z])', r'\1\2', address)

        print(f"Getting coordinates for '{address}'... \t", end='')
        location = geolocator.geocode(address)
        if location:
            cprint("DONE!", Fore.GREEN)
            return (location.latitude, location.longitude)
        cprint('failed', Fore.RED)
    except Exception as e:
        print(e)
    return None


def _crawl_feed(url, city):
    for entry in feedparser.parse(url).entries:
        title, link = entry['title'], entry['link']
        if not db.data[link]:
            prop = parse_page(title, link)
            if prop:
                prop.city = city
                prop.published = datetime(*entry['published_parsed'][:6])
                db.data[link]["property"] = prop


def cprint(msg, color):
    print(color + msg + Style.RESET_ALL)


def _crawl_hemnet():
    cprint("Crawling...", Fore.GREEN)
    for url, city in [
            ('https://www.hemnet.se/mitt_hemnet/sparade_sokningar/15979794.xml', "Lund"),
            ('https://www.hemnet.se/mitt_hemnet/sparade_sokningar/14927895.xml', "Lund"),
            ('https://www.hemnet.se/mitt_hemnet/sparade_sokningar/16190055.xml', "Malmö"),
    ]:
        _crawl_feed(url, city)


def _crawl_afb():
    r = requests.get("https://www.afbostader.se/redimo/rest/vacantproducts")
    data = r.json()['product']
    for a in data:
        p = Property(link=f"https://www.afbostader.se/lediga-bostader/bostadsdetalj/?obj={a['productId']}&area={a['area']}",
                     address=a['address'],
                     area=float(a['sqrMtrs']),
                     monthly_fee=float(a['rent']),
                     rooms=1 if a['shortDescription'] == 'Korridorrum' else float(a['shortDescription'].split()[0].replace(",", ".")),
                     price=0,
                     published=datetime(*map(int, a['reserveFromDate'].split("-"))))
        db.data[p.link]['property'] = p


def crawl():
    _crawl_hemnet()
    _crawl_afb()


def assign_coords(props) -> None:
    cprint("Mapping addresses to coordinates...", Fore.GREEN)
    for prop in props:
        if not prop.coords:
            prop.coords = get_coord(prop.address, prop.city)


def filter_unwanted(props):
    cprint("Filtering away unwanted...", Fore.YELLOW)

    def f(p: Property):
        return p.area >= 55
            # p.monthly_cost() < 12000 and \
            # p.monthly_cost_per_sqm() < 120

    def dist(p: Property):
        if CITY == "Lund":
            return any((p.distance_to(loc) or 0) < r for loc, r in [(lund_c_coords, 2), (lth_coords, 1)])
        elif CITY == "Malmö":
            return min(p.distance_to(loc) or 0 for loc in [malmo_c_coords, triangeln_coords]) < 2.5
        else:
            return True

    return [p for p in props if f(p) and dist(p)]


def main() -> None:
    cprint(f"Looking for apartments in {CITY}", Fore.GREEN)
    crawl()
    db.save()

    props = [v["property"] for v in db.data.values() if "property" in v]
    cprint(f"{len(props)} properties in database", Fore.YELLOW)

    assign_coords(props)
    db.save()

    props = filter_unwanted(props)[-30:]
    # If you want to see AFB apartments, use this filtering instead.
    # props = filter(lambda p: p.price == 0, props)
    props = sorted(props, key=lambda p: p.published)

    cprint(f"Assumptions:\n - Downpayment or loan factor: {LOAN_FACTOR or DOWNPAYMENT}\n - Interest: {INTEREST*100}%\n - Opportunity cost of capital: {OPPORTUNITY_COC_RATE*100}%", Fore.GREEN)
    print(tabulate([p.row() for p in props], headers=Property.headers(), floatfmt=(None, '.0f')))


if __name__ == "__main__":
    db = Datastore()
    main()
