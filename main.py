#!/usr/bin/env python3

import logging
import pathlib
import pickle
from typing import Dict, Any, Tuple, Optional, List
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass

import requests
import feedparser
from colorama import Fore, Style
from geopy import distance
from bs4 import BeautifulSoup
from tabulate import tabulate


logging.basicConfig(level=logging.DEBUG)
logging.getLogger("geopy").setLevel(logging.INFO)
logging.getLogger("urllib3").setLevel(logging.INFO)
log = logging.getLogger(__name__)


lund_c_coords = (55.7068, 13.187)
lth_coords = (55.7124, 13.2091)


class Datastore:
    path = pathlib.Path('./datastore.pickle')
    data: Dict[str, Dict[str, Any]] = defaultdict(lambda: dict())

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

    def __repr__(self):
        return f"<{self.address},\n  Area: {self.area}\n  Rooms: {self.rooms}\n  MoCost: {self.monthly_cost()}\n  MoCost/m²: {self.monthly_cost_per_sqm()}>"

    @staticmethod
    def headers() -> List[str]:
        return ["address", "price", "area", "fee", "rooms", "cost/mo", "cost/m²/mo", "d(Lund C)+d(LTH)", "published"]

    def row(self) -> List[Any]:
        return [
            self.address,
            #self.link,
            self.price,
            self.area,
            self.monthly_fee,
            self.rooms,
            round(self.monthly_cost()),
            round(self.monthly_cost_per_sqm(), 1),
            (self.distance_to(lund_c_coords) or 0) + (self.distance_to(lth_coords) or 0) or None,
            f"{str(self.time_since_published().days) + 'd ago' if self.time_since_published() else ''}",
        ]

    def monthly_cost_per_sqm(self) -> float:
        return self.monthly_cost() / self.area

    def monthly_cost(self, downpayment=800_000, interest=0.016, taxreduction=0.3) -> float:
        return ((self.price - downpayment) * (interest * (1 - taxreduction) / 12)) + self.monthly_fee

    def distance_to(self, other: Tuple[float, float]) -> Optional[float]:
        if not self.coords:
            return None
        return distance.distance(self.coords, other).km

    def time_since_published(self) -> Optional[timedelta]:
        return datetime.now() - self.published if self.published else None


db = Datastore()


def get_entry(link: str) -> str:
    if "page" not in db.data[link]:
        r = requests.get(link)
        with db:
            db.data[link].update(page=r.content)
    else:
        log.debug(f"Got {link} from cache")
    return db.data[link]["page"]


def parse_page(title, link):
    soup = BeautifulSoup(get_entry(link), 'html.parser')

    priceprop = soup.find(class_="property__price")
    if priceprop:
        price = float("".join(priceprop.string.strip().split(" ")[:-1]))
        # print(price)
    else:
        cprint("Probably already sold", Fore.YELLOW)
        return

    monthly_fee = 0
    for p in soup.find(class_="property__attributes").find_all("dd"):
        txt = p.string
        if not txt:
            txt = p.contents[0]
        # &nbsp; to space
        txt = txt.replace("\xa0", " ")
        if "m²" == txt.split()[-1]:
            sqm = float(txt.split(" ")[0].replace(",", "."))
        elif "rum" in txt:
            rooms = float(txt.split(" ")[0].replace(",", "."))
        elif "kr/m" in txt and "n" in txt:
            num = "".join(txt.strip().split(" ")[:-1])
            monthly_fee = float(num.replace(",", "."))

    return Property(link, title, price, sqm, rooms, monthly_fee)


def get_coord(address) -> Optional[Tuple[float, float]]:
    from geopy.geocoders import Nominatim
    geolocator = Nominatim(user_agent="apartmentbuyer")
    location = geolocator.geocode(address + ", Lund")
    if location:
        return (location.latitude, location.longitude)
    return None


def _crawl_feed(url):
    for entry in feedparser.parse(url).entries:
        title, link = entry['title'], entry['link']
        if not db.data[link]:
            prop = parse_page(title, link)
            if prop:
                prop.published = datetime(*entry['published_parsed'][:6])
                db.data[link]["property"] = prop


def cprint(msg, color):
    print(color + msg + Style.RESET_ALL)


def _crawl_hemnet():
    cprint("Crawling...", Fore.GREEN)
    for url in [
        'https://www.hemnet.se/mitt_hemnet/sparade_sokningar/15979794.xml',
        'https://www.hemnet.se/mitt_hemnet/sparade_sokningar/14927895.xml'
    ]:
        _crawl_feed(url)


def _crawl_afb():
    r = requests.get("https://www.afbostader.se/redimo/rest/vacantproducts")
    data = r.json()['product']
    for a in data:
        p = Property(link=f"https://www.afbostader.se/lediga-bostader/bostadsdetalj/?obj={a['productId']}&area={a['area']}",
                     address=a['address'],
                     area=float(a['sqrMtrs']),
                     monthly_fee=float(a['rent']),
                     rooms=1 if a['shortDescription'] == 'Korridorrum' else float(a['shortDescription'].split()[0]),
                     price=0,
                     published=datetime(*map(int, a['reserveFromDate'].split("-"))))
        db.data[p.link]['property'] = p


def crawl() -> List[Property]:
    _crawl_hemnet()
    _crawl_afb()

    assign_coords()
    db.save()
    return [v["property"] for v in db.data.values() if "property" in v]


def assign_coords() -> None:
    cprint("Mapping addresses to coordinates...", Fore.GREEN)
    props = [v["property"] for v in db.data.values() if "property" in v]
    for prop in props:
        if not prop.coords:
            prop.coords = get_coord(prop.address)


def filter_unwanted(props):
    cprint("Filtering away unwanted...", Fore.YELLOW)

    def f(p: Property):
        return p.area >= 55 and \
            p.monthly_cost() < 6000 and \
            p.monthly_cost_per_sqm() < 100 and \
            ((p.distance_to(lund_c_coords) or 0) + (p.distance_to(lth_coords) or 0) < 5)

    return [p for p in props if f(p)]


def main() -> None:
    props = crawl()
    cprint(f"{len(props)} properties in database", Fore.YELLOW)
    props = filter_unwanted(props)
    props = sorted(props, key=lambda p: p.published)

    print(tabulate([p.row() for p in props], headers=Property.headers(), floatfmt=(None, '.0f')))


if __name__ == "__main__":
    main()
