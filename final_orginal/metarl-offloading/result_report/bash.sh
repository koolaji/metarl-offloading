cat tlbo_topo_8.5M.txt  | grep Average | awk '{ print $3 }' | awk '
NR % 5 == 1 && NR <= 100 { min = $1 }  # Start a new group, initialize min
$1 < min && NR <= 100 { min = $1 }      # Update min if a smaller value is found within the first 100 rows
NR % 5 == 0 && NR <= 100 { print min }  # Print min at the end of each group within the first 100 rows
'
min=`cat tlbo_topo_8.5M.txt  | grep Average | awk '{ print $3 }' |awk 'NR == 1 { min = $1 } $1 < min { min = $1 }  END { print min NR  }  '`
echo "min = $min"
#cat org_5.5M.txt  | grep Average | awk '{ print $3 }' | awk '
#NR % 5 == 1 && NR <= 100 { min = $1 }  # Start a new group, initialize min
#$1 < min && NR <= 100 { min = $1 }      # Update min if a smaller value is found within the first 100 rows
#NR % 5 == 0 && NR <= 100 { print min }  # Print min at the end of each group within the first 100 rows
#'
#min=`cat org_5.5M.txt  | grep Average | awk '{ print $3 }' |awk 'NR == 1 { min = $1 } $1 < min { min = $1 }  END { print min NR  }  '`
#echo "min = $min"
