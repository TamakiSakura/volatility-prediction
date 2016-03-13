${YEAR} = $1
ls -1 ${YEAR}.tol > filename
cat ${YEAR}.logvol.+12.txt | awk '{print $1}' > extrafilename
sed 's/$/.mda/' extrafilename > filename2
diff filename filename2
rm filename
rm filename2
rm extrafilename

