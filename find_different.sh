YEAR=$1
ls -1 ${YEAR}.tok > filename
cat ${YEAR}.logvol.+12.txt | awk '{print $2}' > extrafilename
sed 's/$/.mda/' extrafilename > filename2
diff filename filename2
rm filename
rm filename2
rm extrafilename

