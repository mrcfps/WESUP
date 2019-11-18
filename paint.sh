#!/bin/bash

# # GLAS
# python scripts/paint_masks.py records/20190720-0917-PM/results-5scale-new/testA data_glas_all/testA-gt -m wesup -o paintings/wesup-testA-paintings
# python scripts/paint_masks.py records/20190722-1009-AM/results-5scale-new/testA data_glas_all/testA-gt -m sizeloss -o paintings/sizeloss-testA-paintings
# python scripts/paint_masks.py records/20190705-1041-AM/results-new/testA data_glas_all/testA-gt -m cdws -o paintings/cdws-testA-paintings

# python scripts/paint_masks.py records/20190720-0917-PM/results-5scale-new/testB data_glas_all/testB-gt -m wesup -o paintings/wesup-testB-paintings
# python scripts/paint_masks.py records/20190722-1009-AM/results-5scale-new/testB data_glas_all/testB-gt -m sizeloss -o paintings/sizeloss-testB-paintings
# python scripts/paint_masks.py records/20190705-1041-AM/results-new/testB data_glas_all/testB-gt -m cdws -o paintings/cdws-testB-paintings

# CRAG
python scripts/paint_masks.py records/20190713-0348-PM/results-5scale-new CRAG/crag-gt -m wesup -o paintings/wesup-crag-paintings
python scripts/paint_masks.py records/20190726-1246-PM/results-new CRAG/crag-gt -m sizeloss -o paintings/sizeloss-crag-paintings
python scripts/paint_masks.py records/20190717-0905-PM/results-new CRAG/crag-gt -m cdws -o paintings/cdws-crag-paintings

# LUSC
python scripts/paint_masks.py records/20190717-1041-AM/results-5scale-new PD-L1-patches/test/viz -m wesup -o paintings/wesup-lusc-paintings
python scripts/paint_masks.py records/20190726-0226-PM/results-new PD-L1-patches/test/viz -m sizeloss -o paintings/sizeloss-lusc-paintings
python scripts/paint_masks.py records/20190717-1044-AM/results-1scale-new PD-L1-patches/test/viz -m cdws -o paintings/cdws-lusc-paintings