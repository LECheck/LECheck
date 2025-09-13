#!/bin/sh
k=4
m=1
ec_type=isa_l_rs_vand
file_dir=
fragment_dir=

python encode_pyeclib.py -k $k -m $m -ec_type $ec_type -file_dir $file_dir -filenames gpt2-large_wikitext_0_40_full.pth.tar -fragment_dir $fragment_dir