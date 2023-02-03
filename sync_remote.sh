#rsync -chavzP --exclude-from=./rsync_exclude.txt -e "ssh -p 10000" stan@140.112.175.176:~/$1 ./
rsync -chavzP --exclude-from=./rsync_exclude.txt -e "ssh -p 520" stan@140.112.21.58:~/$1 ./
