#awk -F',' 'NR == 1 {print; next} {printf "%s,%s,%s,%f\n", NR,$1,$7,  ($1 - $7)}' progress.csv | tail  -10
#awk -F',' 'NR == 1 {print "Line,", $0; next} {diff = $1 - $7; printf "%d,%s,%s,%f\n", NR, $1,$7, diff; if (diff < min || NR == 2) {min = diff; min_line = $1","$7","($1-$7)}} END {print "Line with minimum difference:", min_line}' progress_3.csv 
awk -F',' 'NR == 1 {print "Line,", $0; next} {diff = $1 - $7; printf "%s,%s,%f\n",  $1,$7, diff; if (diff < min || NR == 2) {min = diff; min_line = $1","$7","($1-$7)}} END {print "Line with minimum difference:", min_line}' progress_3.csv 


