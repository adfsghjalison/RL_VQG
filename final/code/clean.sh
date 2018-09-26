
if [ "$#" -ne 1 ] ; then
	echo "Usage: bash code/clean.sh [dir]"
	exit 1
fi

plan=$1


rm -rf data/all/$plan
rm -rf model/qg/$plan
rm -rf model/final/$plan
rm -rf log/$plan
rm -rf output/$plan

