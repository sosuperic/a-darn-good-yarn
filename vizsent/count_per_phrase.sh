for d in bi_concepts1553/*/ ; do
    echo "$d"
    (cd $d && find . -name '*.jpg' | wc -l)
    #(cd $d && find . -name '*.jpg' | xargs -I {} convert {} -resize "256^>" {})
done
