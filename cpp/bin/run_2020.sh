i=1
max=12
while [ $i -lt $max ]
do
	echo ./main inputs/input-2009-$i-1-To-2009-$((i + 1))-1-.txt
	./main inputs/input-2009-$i-1-To-2009-$((i + 1))-1-.txt &
    true $(( i++ ))
done
