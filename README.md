# nlcd_composite_cli

- nlcd_compositing directory contains original code

nlcd composite cli

## Obsevations

- 2012 did not run for h24 v13 - not sure why

- red band has hotter band values than green or blue QGIS
    - assuming this is the red band

- the Band definitions for GDALINFO could be annotated
    - here is what they look like

```
Band 1 Block=5000x1 Type=UInt16, ColorInterp=Gray
Band 2 Block=5000x1 Type=UInt16, ColorInterp=Undefined
Band 3 Block=5000x1 Type=UInt16, ColorInterp=Undefined
Band 4 Block=5000x1 Type=UInt16, ColorInterp=Undefined
Band 5 Block=5000x1 Type=UInt16, ColorInterp=Undefined
Band 6 Block=5000x1 Type=UInt16, ColorInterp=Undefined
```

# Todo 

- write pc_compsite - parallel cluster composite
- shoriz svert - ehoriz evert
- start year - end year
- ask for all cpus 8 how with salloc syntax?
