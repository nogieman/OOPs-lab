#!/usr/bin/bash

script_dir="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd $script_dir/..
git submodule update --init --recursive --depth 1 -j "$(nproc --all)"

cd third_party
# install boost
boost_ver1="1.86.0"
boost_ver2="$(echo $boost_ver1 | sed 's/\./_/g')"
tar_file="boost_$boost_ver2.tar.gz"
wget "https://archives.boost.io/release/$boost_ver1/source/$tar_file"
tar --one-top-level=boost -xzf $tar_file
mv boost/boost_"$boost_ver2"/* boost/

rm -rf boost/boost_"$boost_ver2" *.tar.gz
