#!/user/bin/env python
import csv
import re
# with open('/Users/nancysackman/Code/building/attributeCollapse.csv','r') as csvfile:
building_reader=csv.reader(open("/Users/nancysackman/Code/building/attColl.csv"))
next(building_reader, None)
arr_out = []
for row in building_reader:
    # print(', '.join(row))
    # print(row[3])
    latitud_str=re.search('lat\D*: -?\d*\.\d*', row[3]) #first two rows were headers and skipped
    if latitud_str:
        lat = re.search('-?\d.*\.\d*', latitud_str.group())
        if lat:
            lat = round(float(lat.group()),5)
            print(lat)
    # lat_str=re.search('lat: -?\d*\.?\d*',row[3])
    # if lat_str:
    #     lat1=re.search('-?\d.*\.?\d*', lat_str.group())
    #     if lat1:
    #         lat1 = round(float(lat1.group()),5)
    #         print(lat1)
    #         dict_out['lat']=lat1


    longitud_str=re.search('lon[:g]\D*:? -?\d*\.\d*', row[3])
    if longitud_str:
        lon = re.search('-?\d.*\.\d*', longitud_str.group())
        if lon:
            lon = round(float(lon.group()),5)
            print(lon)


    # lon_str=re.search('lon: -?\d*\.?\d*', row[3])
    # if lon_str:
    #     lon1=re.search('-?\d.*\.?\d*', lon_str.group())
    #     if lon1:
    #         lon1 = round(float(lon1.group()),5)
    #         print(lon1)
    #         dict_out['lon']=lon1

    match=False
    for coor in arr_out:
        if lat==coor['lat'] and lon==coor['lon']:
            match=True
            break
    if not match:
        arr_out.append({'lat':lat,'lon':lon})
print(arr_out)

#Writing CSV File
# need to compare lats and lons with each other due to duplicates


with open('attributeCollapse.csv', mode='w') as csv_out:
    fieldnames=['lat','lon']
    writer=csv.DictWriter(csv_out,fieldnames=fieldnames)
    writer.writeheader()
    for row in arr_out:
        writer.writerow(row)
