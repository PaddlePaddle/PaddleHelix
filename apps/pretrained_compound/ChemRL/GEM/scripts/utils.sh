#!/bin/bash

############
# use ctrl+c to kill parallel exec
############
trap killgroup SIGINT
killgroup(){
  echo killing...
  kill -9 0
}

get_last_epoch() {
    local epoch_prefix=$1
    for ((i=1000;i>0;i-=2)); do
        if [ -d "${epoch_prefix}$i" ]; then
            echo $i
            return
        fi
    done
}

MARK() {
    local name=$1
    touch $name.DONE
}

CHECK() {
    local name=$1
    if [ ! -f $name.DONE ]; then
        return 1
    fi
    return 1
}

ASSERT() {
    local name=$1
    if [ ! -f $name.DONE ]; then
        echo "[FAILED] $name failed."
        exit 1
    fi
}

pull_if_exists() {
    local hadoop_model_dir=$1
    local model_dir=$2
    $HADOOP_FS -test -e $hadoop_model_dir/.done
    if [ $? -eq 0 ]; then
        echo "$hadoop_model_dir/.done already exists."
        mkdir -p $model_dir
        cd $model_dir
        $HADOOP_FS -get $hadoop_model_dir/epoch*.tgz .
        files=`ls epoch*.tgz`
        for f in $files; do
            tar xzf $f
        done
        cd -
        return 0
    fi
    return 1
}

push() {
    local model_dir=$1
    local hadoop_model_dir=$2
    if [ ! -d "$model_dir/epoch_best" ]; then
        echo "[FAILED] cant push, ($model_dir/epoch_best) not exists."
        exit 1
    fi
    cd $model_dir
    files=`ls | grep epoch`
    for f in $files; do
        tar czf $f.tgz $f
    done
    $HADOOP_FS -rmr $hadoop_model_dir/epoch*.tgz
    $HADOOP_FS -rmr $hadoop_model_dir/.done
    $HADOOP_FS -mkdir $hadoop_model_dir
    $HADOOP_FS -put epoch*.tgz $hadoop_model_dir/
    $HADOOP_FS -touchz $hadoop_model_dir/.done
    $HADOOP_FS -ls $hadoop_model_dir
    echo "Upload to $hadoop_model_dir"
    cd -
}
