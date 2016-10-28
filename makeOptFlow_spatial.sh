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
   echo "Usage: ./makeOptFlow_spatial <filePattern> <outputFolder> [<startNumber>]"
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
    frame_l_file = "${folderName}/${frame_basename}_l.ppm"
    frame_r_file = "${folderName}/${frame_basename}_r.ppm"
    flow_lr_file = "${folderName}/${frame_basename}_lr.flo"
    flow_rl_file = "${folderName}/${frame_basename}_rl.flo"

		eval "ruby" "cut_l_r.rb" "$frame" "$frame_l_file" "$frame_r_file"

    if [ ! -f $flow_lr_file ]; then
      eval $flowCommandLine "$frame_l_file" "$frame_r_file" "$flow_lr_file"
    fi
    if [ ! -f $flow_rl_file ]; then
      eval $flowCommandLine "$frame_r_file" "$frame_l_file" "$flow_rl_file"
    fi
    ./consistencyChecker/consistencyChecker "$flow_lr_file" "$flow_rl_file" "${folderName}/reliable_${frame_basename}_lr.pgm"
    ./consistencyChecker/consistencyChecker "$flow_rl_file" "$flow_lr_file" "${folderName}/reliable_${frame_basename}_rl.pgm"
  else
    break
  fi
  i=$[$i +1]
done
