progressfilt ()
{
    local flag=false c count cr=$'\r' nl=$'\n'
    while IFS='' read -d '' -rn 1 c
    do
        if $flag
        then
            printf '%s' "$c"
        else
            if [[ $c != $cr && $c != $nl ]]
            then
                count=0
            else
                ((count++))
                if ((count > 1))
                then
                    flag=true
                fi
            fi
        fi
    done
}


FILE=$1

if [ $FILE == "dataset-VCC" ]; then
    # VCC dataset including 4 speakers
    URL=http://www.kecl.ntt.co.jp/people/kameoka.hirokazu/data/mvae/vcc.zip
    mkdir -p ../data/
    ZIP_FILE=../data/vcc.zip
    wget --progress=bar:force $URL -O $ZIP_FILE 2>&1 | progressfilt
    unzip -qq $ZIP_FILE -d ../data/
    rm $ZIP_FILE

elif [ $FILE == "test-samples" ]; then
    # test samples for VCC dataset
    URL=http://www.kecl.ntt.co.jp/people/kameoka.hirokazu/data/mvae/test_input.zip
    mkdir -p ../data/
    ZIP_FILE=../data/test_input.zip
    wget --progress=bar:force $URL -O $ZIP_FILE 2>&1 | progressfilt
    unzip -qq $ZIP_FILE -d ../data/
    rm $ZIP_FILE

elif [ $FILE == "models" ]; then
    # pretrained models 
    URL=http://www.kecl.ntt.co.jp/people/kameoka.hirokazu/data/mvae/models.zip
    mkdir -p ../model/
    ZIP_FILE=../model/models.zip
    wget --progress=bar:force $URL -O $ZIP_FILE 2>&1 | progressfilt
    unzip -qq $ZIP_FILE -d ../model/
    mv ../model/models/ ../model/pretrained_model/
    rm $ZIP_FILE

else
    echo "Available arguments are dataset-VCC, test-samples, models."
    exit 1
​
fi
