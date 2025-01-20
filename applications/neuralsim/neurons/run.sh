
j=0
while ((j<1000))
do
   python test.py  #>> liftest.log
#   python test_Adex.py  >> Adextest.log
#  python test_izhikevich.py >>izhikevich.log
   echo "times is $j"
  j=$(($j+1))  
done
