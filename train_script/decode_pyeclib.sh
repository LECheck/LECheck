#!/bin/sh
k=4
m=1
ec_type=isa_l_rs_vand
file_dir=
filenumber=4
fragment_dir=
output_path=
filenames=

python decode_pyeclib.py -k $k -m $m -ec_type $ec_type -file_dir $file_dir -filenames gpt2-large_wikitext_0_10_full.pth.tar \
    -filenumber $filenumber -fragment_dir $fragment_dir -output_path $output_path