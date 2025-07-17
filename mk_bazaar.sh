DIR=BAZAAR
mkdir -p $DIR
for n in `seq 10 10 190` `seq 200 100 1000` `seq 2000 1000 10000`; do
    echo -n -e "n: $n\t";
    for i in `seq 25`; do
        python3 mk_bazaar.py $i $n > $DIR/bazaar-$n-$i.txt
    done
done
