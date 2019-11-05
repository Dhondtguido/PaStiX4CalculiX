#!/bin/sh

if [ $# -gt 0 ]
then
    BUILDDIR=$1
fi
BUILDDIR=${BUILDDIR-=build}

echo $PWD
rm -f filelist.txt

git ls-files | grep "\.[ch]"   >  filelist.txt
#git ls-files | grep "\.py"     >> filelist.txt
find $BUILDDIR -name '*\.[ch]' >> filelist.txt
#echo "wrappers/python/examples/pypastix/enum.py" >> filelist.txt

# Remove all CMakeFiles generated files
sed -i '/CMakeFiles/d' filelist.txt

# Remove installed files
sed -i '/^install.*/d' filelist.txt

# Remove original files used for precision generation
for file in `git grep "@precisions" | awk -F ":" '{ print $1 }'`
do
    sed -i "\:^$file.*:d" filelist.txt
done

# Remove external header files
for file in include/cblas.h include/lapacke.h
do
    sed -i "\:^$file.*:d" filelist.txt
done

# Remove external driver files
for file in src/drivers/iohb.c src/drivers/iohb.h src/drivers/mmio.c src/drivers/mmio.h
do
    sed -i "\:^$file.*:d" filelist.txt
done
