#!/usr/bin/python3

from __future__ import print_function
import csv
import re
import os.path

fieldnames = [ 'id', 'time', 'name', 'thread', 'value' ]

csvfile = open("/tmp/counters.csv")
reader = csv.DictReader(csvfile, fieldnames=fieldnames, delimiter=';')

data = []
for row in reader:
    data.append( row )

def keycmp( r ):
    return float( r['time'] )

data.sort( key=keycmp )

counter = []
entry = { 'time' : 0.,
          'value' : 0 }
for d in data:
    t = float(d['time'])
    if ( t < entry['time'] ):
        print("Error %e > %e"%( t, entry['time'] ) )
    elif ( t > entry['time'] ):
        counter.append( entry.copy() )
        entry['time']  = t
        entry['value'] = 0

    val = int(float(d['value']))
    if int(d['id']) == 31:
        entry['value'] = entry['value'] + val
    elif int(d['id']) == 32:
        entry['value'] = entry['value'] - val

counter.append( entry )

d = data[0]
c = counter[0]
gvalue = c['value']
print( "%d %e %s %s %e" % ( 30, c['time'], d['name'], d['thread'], c['value'] ) )
for c in counter[1:]:
    if c['value'] > 0:
        i = 31
        v = c['value']
    elif c['value'] < 0:
        i = 32
        v = - c['value']
    else:
        continue

    gvalue = gvalue + c['value']
    if gvalue < 0 :
        print( "gvalue=%d" % ( gvalue ) )
    print( "%d %e %s %s %e" % ( i, c['time'], d['name'], d['thread'], v ) )

