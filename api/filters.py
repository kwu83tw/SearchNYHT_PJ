from collections import Counter
from geopy.distance import geodesic
from geopy.geocoders import Nominatim

import abc
import math
import pandas as pd
import re
# Table columns
LAT = "latitude"
LONGT = "longitude"
DIST = "distance"
ID = "id"
ROOMTYPE = "room_type"
PRICE = "price"
MINNIGHTS = "minimum_nights"
NUMREVIEWS = "number_of_reviews"
LASTREVIEW = "last_review"
REVIEWPM = "reviews_per_month"
AVAL = "availability_365"
# TODO: to update more attractions
ATTRACTIONS = ["manhattan skyline",
               "central park",
               "brooklyn bridge",
               "grand central terminal",
               "empire state building",
               "statue of liberty",
               "time square",
               "rockefeller center",
               "radio city music hall",
               "madison square garden"]
WORD = re.compile(r'\w+')
TH = 0.8


class Filter(metaclass=abc.ABCMeta):
    """Base filter class"""
    csv_path, wt = None, []
    baseDfs, filtered_dfs = pd.DataFrame(), pd.DataFrame()
    metadata = {}

    def prep(self, cols=[]):
        """
        Prepare the data.
        """
        self.filtered_dfs = pd.read_csv(self.csv_path, usecols=cols)
        return self.filtered_dfs

    @abc.abstractmethod
    def process(self):
        """
        The logic to process the data and return interested DataFrame portion.
        """
        pass

    def set_csvpath(self, path):
        self.csv_path = path
        return self

    def set_baseDfs(self, basedfs):
        self.baseDfs = basedfs
        return self

    def set_wt(self, wt):
        self.wt = wt
        return self

    def set_metadata(self, **kwargs):
        for k, v in kwargs.items():
            self.metadata[k] = v

    def merge_datasets(self):
        dfs = ljoin_dataframe(self.baseDfs, self.dfs)
        self.set_baseDfs(dfs)
        return dfs


class LandmarkFilter(Filter):
    latitude, longitude, distance = 0.0, 0.0, 0.0

    def prep(self, cols=[ID, "latitude", "longitude"]):
        self.dfs = pd.read_csv(self.csv_path, usecols=cols)
        self.dfs["coord"] = self.dfs.apply(lambda row: self._get_coord(row),
                                           axis=1)
        self.dfs = self.dfs[self.dfs["coord"] != -1]
        self._set_given_coord()
        return self.dfs

    def _calculate_dist_from_coord(self, target):
        self.dfs["dist_to_target"] = self.dfs["coord"].apply(lambda x: f"{self._get_dist_from_targets(x, target)}")

    def process(self):
        target = self._get_nearby()
        if target and len(target) == 2:
            self.latitude, self.longitude = target[0], target[1]
        else:
            target = (self.latitude, self.longitude)
        self._set_distance()
        #TODO: Need to speed up apply()
        self._calculate_dist_from_coord(target)
        #    tmp = tempfile.NamedTemporaryFile()
        #    dfs.to_csv(tmp.name, index=False)
        #    tmp.close()
        # distance less than user query dist
        self.dfs["dist_to_target"] = self.dfs["dist_to_target"].astype("float")
        less_than_dist = self.dfs["dist_to_target"] <= float(self.distance)
        self.dfs = self.dfs[less_than_dist]
        return self.dfs

    def merge_datasets(self):
        self.set_baseDfs(self.dfs)
        return self.dfs

    def _get_coord(self, row):
        lat = row[LAT]
        longt = row[LONGT]
        try:
            lat = float(lat)
            longt = float(longt)
        except ValueError:
            return -1
        return str(lat) + ',' + str(longt)

    def _set_given_coord(self):
        self.latitude, self.longitude = self.metadata.get(LAT), self.metadata.get(LONGT)

    def _set_distance(self):
        self.distance = self.metadata.get(DIST)

    def _get_nearby(self):
        """
        Get the popular landmark to coordinates
        Refer Cosine similarity to compare strings and get coordinates if it's
        above threshold.
        """
        if "near" in self.wt or "close" in self.wt:
            landmark = ' '.join(self.wt[1:])
            for lm in ATTRACTIONS:
                cosine = self._get_cosine(self._text_to_vector(landmark),
                                    self._text_to_vector(lm))
                if cosine >= TH:
                    addrs = "%s, NY, USA" % lm
                    locator = Nominatim(user_agent="myGG")
                    loc = locator.geocode(addrs)
                    return (loc.latitude, loc.longitude)
        return

    def _get_dist_from_targets(self, t1, t2):
        """
        Get distance between two coordinates
        :return: Distance in metres
        """
        #t2 = (row[LAT], row[LONGT])
        t1 = (float(t1.split(',')[0]), float(t1.split(',')[1]))
        return geodesic(t1, t2).m

    # reference:
    # https://stackoverflow.com/questions/15173225/calculate-cosine-similarity-given-2-sentence-strings
    def _get_cosine(self, vec1, vec2):
        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum([vec1[x] * vec2[x] for x in intersection])

        sum1 = sum([vec1[x]**2 for x in vec1.keys()])
        sum2 = sum([vec2[x]**2 for x in vec2.keys()])
        denominator = math.sqrt(sum1) * math.sqrt(sum2)

        if not denominator:
            return 0.0
        else:
            return float(numerator) / denominator

    def _text_to_vector(self, text):
        words = WORD.findall(text)
        return Counter(words)


class ReviewFilter(Filter):
    def prep(self,
             cols=[ID,
                   NUMREVIEWS,
                   LASTREVIEW,
                   REVIEWPM]):
        self.dfs = pd.read_csv(self.csv_path, usecols=cols)
        return self.dfs

    def process(self):
        pass


class MinimalstayFilter(Filter):
    def prep(self,
             cols=[ID,
                   MINNIGHTS]):
        self.dfs = pd.read_csv(self.csv_path, usecols=cols)
        return self.dfs

    def process(self):
        pass


class AvailabilityFilter(Filter):
    def prep(self,
             cols=[ID,
                   AVAL]):
        self.dfs = pd.read_csv(self.csv_path, usecols=cols)
        return self.dfs

    def process(self):
        pass


class PriceFilter(Filter):
    def prep(self, cols=[ID, PRICE]):
        self.dfs = pd.read_csv(self.csv_path, usecols=cols)
        return self.dfs

    def process(self):
        pass


class RoomFilter(Filter):
    room_type = []

    def prep(self, cols=[ID, ROOMTYPE]):
        self.dfs = pd.read_csv(self.csv_path, usecols=cols)
        self._get_room_type()
        return self.dfs

    def process(self):
        if len(self.room_type) == 1:
            self.dfs = self.dfs[self.dfs[ROOMTYPE] == self.room_type[0]]
        elif len(self.room_type) == 2:
            self.dfs = self.dfs[self.dfs[ROOMTYPE] == self.room_type[0] & self.dfs[ROOMTYPE] == self.room_type[-1]]
        return self.dfs

    def _get_room_type(self):
        """
        Available room type are "Entire home/apt", "Private room", "Shared
        room". Return list of room type.
        """
        abr = ["Entire home/apt", "Private room", "Shared room"]
        if "bedroom" in self.wt or "room" in self.wt:
            if "one" in self.wt or "1" in self.wt:
                self.room_type = abr[1:]
            else:
                self.room_type = [abr[0]]
        return self


def ljoin_dataframe(dfs1, dfs2, key="id"):
    return pd.merge(dfs1, dfs2, on=key)
