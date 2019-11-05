#!/bin/bash

changelog=""

function gen_changelog()
{
    local firstline=$( grep -n "^# " ChangeLog.md | head -n 1 | cut -d ':' -f 1 )
    firstline=$(( firstline + 1 ))
    #echo $firstline
    local lastline=$( grep -n "^# " ChangeLog.md | head -n 2 | tail -n 1 | cut -d ':' -f 1 )
    lastline=$(( lastline - 1 ))
    #echo $lastline

    for i in `seq $firstline $lastline`
    do
        local line=$( head -n $i ChangeLog.md | tail -n 1 )
        changelog="$changelog$line\\n"
        #echo $line
    done
}

release=""
function get_release()
{
    local firstline=$( grep -n "^# " ChangeLog.md | head -n 1 | cut -d ':' -f 1 )
    release=$( head -n $firstline ChangeLog.md | tail -n 1 | sed 's/# pastix\-//' )
}

# Get the release name through the branch name, and through the ChangeLog file.
# Both have to match to be correct
RELEASE_NAME=`echo $CI_COMMIT_REF_NAME | cut -d - -f 2`
get_release

if [ -z "$RELEASE_NAME" -o -z "$release" -o "$RELEASE_NAME" != "$release" ]
then
    echo "Commit name $RELEASE_NAME is different from ChangeLog name $release"
    exit 1
fi

wget https://raw.githubusercontent.com/Kentzo/git-archive-all/master/git_archive_all.py
mv git_archive_all.py git-archive-all
chmod +x git-archive-all
./git-archive-all --force-submodules pastix-$RELEASE_NAME.tar.gz

GETURL=`echo curl --request POST --header \"PRIVATE-TOKEN: $RELEASE_TOKEN\" --form \"file=\@pastix-$RELEASE_NAME.tar.gz\" https://gitlab.inria.fr/api/v4/projects/$CI_PROJECT_ID/uploads`
MYURL=`eval $GETURL | jq .url | tr -d '"'`
# could be replaced by : | cut -d , -f 2 | cut -d : -f 2
# new tag "v$RELEASE_NAME" should have been commited

# extract the change log from ChangeLog.md
gen_changelog

# Try to remove the release if it already exists
curl --request DELETE --header "PRIVATE-TOKEN: $RELEASE_TOKEN" https://gitlab.inria.fr/api/v4/projects/$CI_PROJECT_ID/releases/v$RELEASE_NAME

# Generate the curl command that create the release
COMMAND=`echo curl --header \"Content-Type: application/json\" --header \"PRIVATE-TOKEN: $RELEASE_TOKEN\" \
  --data \'{ \"name\": \"v$RELEASE_NAME\", \
            \"tag_name\": \"v$RELEASE_NAME\", \
            \"ref\": \"$CI_COMMIT_REF_NAME\", \
            \"description\": \"$changelog\", \
            \"assets\": { \"links\": [{ \"name\": \"Download release\", \"url\": \"$CI_PROJECT_URL/$MYURL\" }] } }\' \
  --request POST https://gitlab.inria.fr/api/v4/projects/$CI_PROJECT_ID/releases`
eval $COMMAND
