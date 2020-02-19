# Assumption:
* Only support single purpose query string for now, e.g., "near by empire state building", "two bedrooms". No support on query string like "two bedrooms, and near empire state building".


# Step to run the code:
My test m/c is on Ubuntu 18.04, python 3.6.8
* Create virtualenv first 
>
> virtualenv myenv


* Activate virtualenv and install needed packages (You can compare your pip
  freeze -l output to tell if any difference.)
>
> source ./myenv/bin/activate
>
>
> pip install bottle pandas geopy nltk curl folium 
>


* Activating the API
>
> python main.py
>

* curl a GET request to see if everything is okay on other consle.

```
    kwu@kwu-desktop:~$ curl -v http://127.0.0.1:8099/test
    *   Trying 127.0.0.1...
    * TCP_NODELAY set
    * Connected to 127.0.0.1 (127.0.0.1) port 8099 (#0)
    > GET /test HTTP/1.1
    > Host: 127.0.0.1:8099
    > User-Agent: curl/7.58.0
    > Accept: */*
    >
    * HTTP 1.0, assume close after body
    < HTTP/1.0 200 OK
    < Date: Sat, 18 Jan 2020 07:28:18 GMT
    < Server: WSGIServer/0.2 CPython/3.6.8
    < Content-Type: application/json
    < Content-Length: 21
    <
    * Closing connection 0
    {"status": "success"}
```

* If curl command works fine, then you shall be able to query stockName.

```
curl -v --header "Content-Type:application/json" http://127.0.0.1:8099/search --data '{"latitude" : "41", "longitude": "-73", "distance": "300.7", "query": "near the empire state building"}'
```


# The searchNYHT project 

* Make use of Pandas Dataframe to deal with dataset extracting from csv.
* Use virtualenv.
* Hands on geopy and ntlk library to learn new thing


# Future improvement
* More refined filters
* Parallelize the process for heavy compute task (dataframe computing on distance). The API reponse time is unacceptable now and might break due to heavy request.
* Make use of geopandas
* More refined logic on getting information from user query string.
* More validation works.
* Too many to be addressed .... ><
