from bottle import request, response
from bottle import post, get, put, delete
from collections import Counter
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from .filters import LandmarkFilter, RoomFilter, ReviewFilter
from .filters import MinimalstayFilter, AvailabilityFilter, PriceFilter

import datetime as DT
import dateutil
import folium
import json
import math
import nltk
import pandas as pd
import re
import string
import tempfile


LAT = "latitude"
LONGT = "longitude"
DIST = "distance"
WORDTOKENS = "wordtokens"
QUERY_STR = "query"
CSV_PATH = "./AB_NYC_2019.csv"
NY_CITY_COORD = [40.730610, -73.935242]
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


def _validation(func):
    """Validation decorator"""
    def wrapper(*args, **kwargs):
        return eval('_validate_' + func.__name__)(*args, **kwargs)
    return wrapper


@get('/test')
def listen_handler():
    response.headers['Content-Type'] = 'application/json'
    return json.dumps({'status': 'success'})


#def draw_map(dfs, lat, longt):
#    ny_map = folium.Map(location=NY_CITY_COORD,
#                        zoom_start=12,
#                        tiles="cartodbpositron")
#    # User given coord
#    folium.Marker([lat, longt]).add_to(ny_map)
#    # HT coords
#    dfs.apply(lambda row: folium.CircleMarker(location=[row[LAT], row[LONGT]],
#                                             popup=row["price"]).add_to(ny_map),
#                                             axis=1)
#    return ny_map._repr_html_()


def reformat_qstring(qs):
    """
    Make use of NTLK library to cleanup the query string and resturn a list of
    word tokens.
    """
    if len(qs) == 0:
        return

    qs.lower()
    # Remove punctuations
    res = qs.translate(str.maketrans("", "", string.punctuation))
    # Remove space
    res = res.strip()
    # Remove stopwords
    nltk.download("stopwords")
    sw = set(stopwords.words("english"))
    nltk.download('punkt')
    tokens = word_tokenize(res)
    res = [e for e in tokens if e not in sw]
    # lemmatiation
    nltk.download('wordnet')
    lemmatizer = WordNetLemmatizer()
    res = [lemmatizer.lemmatize(w) for w in res]
    return res


class FilterManager(object):
    target_lat, target_longt, target_dist = 0.0, 0.0, 0.0
    dfs, baseDfs = pd.DataFrame(), pd.DataFrame()

    def __init__(self, filters=[], **kwargs):
        self.kwargs = kwargs
        self.filters = filters
        self.target_lat = kwargs.get(LAT)
        self.target_longt = kwargs.get(LONGT)
        self.target_dist = kwargs.get(DIST)

    def execute(self):
        for ft in self.filters:
            #import pdb; pdb.set_trace()
            ftp = ft()
            ftp.set_wt(self.kwargs.get(WORDTOKENS))
            ftp.set_metadata(**self.kwargs)
            # TODO: Construct a data container instead of passing it in and out
            # of filters to avoid sophisticated manipulations
            # Break if no result found within given distance(LandmarkFilter)
            if ft.__name__ != "LandmarkFilter" and self.baseDfs.empty:
                return self.dfs
            else:
                ftp.set_baseDfs(self.baseDfs)
            ftp.set_csvpath(CSV_PATH).prep()
            ftp.process()
            dfs = ftp.merge_datasets()
            self._update_metadata(ftp)
        self.dfs = dfs
        return dfs

    def _update_metadata(self, ft):
        if hasattr(ft, LAT):
            self.target_lat = getattr(ft, LAT)
        if hasattr(ft, LONGT):
            self.target_longt = getattr(ft, LONGT)
        if hasattr(ft, DIST):
            self.target_dist = getattr(ft, DIST)
        if hasattr(ft, 'baseDfs'):
            self.baseDfs = getattr(ft, 'baseDfs')


@post('/search')
def req_handler():
    """
    Request handler
    """
    try:
        try:
            data = request.json
        except Exception:
            raise ValueError

        if data is None:
            raise ValueError

        try:
            # Required fields
            lat = data.get(LAT)
            longt = data.get(LONGT)
            dist = data.get(DIST)
            if not lat or not longt or not dist:
                raise KeyError
            # optional
            query_str = data.get(QUERY_STR)

        except (TypeError, KeyError):
            # indicate that user doesn't provide proper body
            raise ValueError

        word_tokens = reformat_qstring(query_str)
        filters = [LandmarkFilter, RoomFilter, ReviewFilter,
                   MinimalstayFilter, AvailabilityFilter, PriceFilter]
        fm = FilterManager(filters,
                           **{LAT: lat,
                              LONGT: longt,
                              DIST: dist,
                              WORDTOKENS: word_tokens})
        dfs = fm.execute()
        #import pdb; pdb.set_trace()
        if dfs.empty:
            res = "Cannot find any place within given coordinate and distance"
        else:
            dfs.drop([LAT, LONGT, "coord"], axis=1)
            dfs.sort_values(by=["price"])
            res = dfs.to_dict("records")

    except ValueError:
        # if bad request data, return 400 Bad Request
        response.status = 400
        return

    except KeyError:
        response.status = 409
        return

    response.headers['Content-Type'] = 'text/json'
    response.status = 200
    return json.dumps({
        "success": "true",
        LAT: fm.target_lat,
        LONGT: fm.target_longt,
        QUERY_STR: query_str,
        "result": res
        })
