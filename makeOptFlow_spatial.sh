# Specify the path to the optical flow utility here.
# Also check line 44 and 47 whether the arguments are in the correct order.
flowCommandLine="bash run-deepflow.sh"

if [ -z "$flowCommandLine" ]; then
  echo "Please open makeOptFlow.sh and specify the command line for computing the optical flow."
  exit 1
fi

if [ ! -f ./consistencyChecker/consistencyChecker ]; then
  if [ ! -f ./consistencyChecker/Makefile ]; then
    echo "Consistency checker makefile not found."
    exit 1
  fi
  cd consistencyChecker/
  make
  cd ..
fi

filePattern=$1
folderName=$2
startFrame=${3:-1}

if [ "$#" -le 1 ]; then
   echo "Usage: ./makeOptFlow <filePattern> <outputFolder> [<startNumber> [<stepSize>]]"
   echo -e "\tfilePattern:\tFilename pattern of the frames of the videos."
   echo -e "\toutputFolder:\tOutput folder."
   echo -e "\tstartNumber:\tThe index of the first frame. Default: 1"
   exit 1
fi

i=$[$startFrame]

mkdir -p "${folderName}"

while true; do
  frame=$(printf "$filePattern" "$i")
  if [ -a $frame ]; then
		frame_basename=$(basename -s .ppm $frame)
		eval "ruby" "cut_l_r.rb" "$frame" "${frame_basename}_l.ppm" "${frame_basename}_r.ppm"
    if [ ! -f ${folderName}/${frame_basename}_lr.flo ]; then
      eval $flowCommandLine "${frame_basename}_l.ppm" "${frame_basename}_r.ppm" "${folderName}/${frame_basename}_lr.flo"
    fi
    if [ ! -f ${folderName}/${frame_basename}_rl.flo ]; then
      eval $flowCommandLine "${frame_basename}_r.ppm" "${frame_basename}_l.ppm" "${folderName}/${frame_basename}_rl.flo"
    fi
    ./consistencyChecker/consistencyChecker "${folderName}/${frame_basename}_lr.flo" "${folderName}/${frame_basename}_rl.flo" "${folderName}/reliable_${frame_basename}_lr.pgm"
    ./consistencyChecker/consistencyChecker "${folderName}/${frame_basename}_rl.flo" "${folderName}/${frame_basename}_lr.flo" "${folderName}/reliable_${frame_basename}_rl.pgm"
  else
    break
  fi
  i=$[$i +1]
done
