#awk -F',' 'NR == 1 {print; next} {printf "%s,%s,%s,%f\n", NR,$1,$7,  ($1 - $7)}' progress.csv | tail  -30
awk -F','  -v one="$1" -v two="$2"  'NR == 1 {print "Line,", $0; next} {diff = $one - $two; printf "%d,%s,%s,%f\n", NR, $one,$two, diff; if (diff < min || NR == 2) {min = diff; min_line = $one","$tow","($one-$two)}} END {print "Line with minimum difference:", min_line}' progress.csv | tail -30 
#awk -F','  -v one="$1" -v two="$2"  'NR == 1 {print "Line,", $0; next} {diff = $one - $two; printf "%d,%s,%s,%f\n", NR, $one,$two, diff; if (diff < min || NR == 2) {min = diff; min_line = $one","$tow","($one-$two)}} END {print "Line with minimum difference:", min_line}' progress_7.csv | tail -30 


