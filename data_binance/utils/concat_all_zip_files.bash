root=~/binance_data
csv_root=~/binance_data/csv_data
cd $root
mkdir -p $csv_root
echo "Processing FUTURES data"
for symbol in $(ls ~/binance_data/data/futures/um/monthly/klines); do
    echo "process $symbol data ..."
    monthly_format=data/futures/um/monthly/klines/$symbol/1m/
    daily_format=data/futures/um/daily/klines/$symbol/1m/
    file_name="$symbol-1m-perpetual.csv"
    
    pushd $monthly_format
    unzip -qn "*.zip"
    rm -f "$csv_root/$file_name"
    
    for f in $(ls ./*.csv); do
        head -1 $f | grep open &> /dev/null
        if [[ $? == 0 ]]; then
            # echo "$f detected header"
            # echo "$(head -1 $f)"
            tail -n +2 -q $f >> "$csv_root/$file_name"
        else
            cat $f >> "$csv_root/$file_name"
        fi
    done
    # cat *.csv > "$csv_root/$file_name"
    popd

    pushd $daily_format
    unzip -qo "*.zip"
    for f in $(ls ./*.csv); do
        head -1 $f | grep open &> /dev/null
        if [[ $? == 0 ]]; then
            # echo "$f detected header"
            # echo "$(head -1 $f)"
            tail -n +2 -q $f >> "$csv_root/$file_name"
        else
            cat $f >> "$csv_root/$file_name"
        fi
    done
    popd
done

# echo "Processing SPOT data"
# for symbol in $(ls /Users/bytedance/binance_data/data/spot/monthly/klines); do
#     echo "process $symbol data ..."
#     monthly_format=data/spot/monthly/klines/$symbol/1m/
#     daily_format=data/spot/daily/klines/$symbol/1m/
#     file_name="$symbol-1m.csv"

#     pushd $monthly_format
#     unzip -qo "*.zip"
#     rm -f "$csv_root/$file_name"
#     cat *.csv > "$csv_root/$file_name"
#     popd

#     pushd $daily_format
#     unzip -qo "*.zip"
#     cat *.csv >> "$csv_root/$file_name"
#     echo "process $symbol data to $csv_root/$file_name"
#     popd

# done

# # concat long-short data
# for symbol in $(ls /Users/bytedance/binance_data/data/futures/ls); do
#     echo "process $symbol data ..."
#     path=data/futures/ls/$symbol
#     file_name="$symbol-5m-ls.csv"
#     pushd $path
#         cat $symbol-5m-ls-* > "$csv_root/$file_name"
#     popd
# done





# # cat "$monthly_format/daily_bar.csv" "$daily_format/monthly_bar.csv" > "$symbol_bar.csv"