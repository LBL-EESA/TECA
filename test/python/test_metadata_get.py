import teca

md = teca.teca_metadata()
md['present'] = 5 

assert md.get('present') == 5      #1-arg, key present
assert md.get('present', 42) == 5  #2-arg, key present -> value
assert md.get('missing', 42) == 42 #2-arg, key missing -> default
try:
   md.get('missing')               #1-arg, key missing
   raise AssertionError('expected KeyError')
except KeyError:
   pass

print('ok')
