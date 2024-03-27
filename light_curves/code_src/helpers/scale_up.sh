# For complete usage information, see the scale_up notebook tutorial.
HELPER_PY="$(dirname "$0")/scale_up.py"
RUN_HELPER_PY(){
    build=$1
    archive=$2

    if [ $archive ]; then
        nohup python $HELPER_PY --build $build --archive $archive --kwargs_json "$kwargs_json" --kwargs_dict ${kwargs_dict[@]} &
    else
        python $HELPER_PY --build $build --kwargs_json "$kwargs_json" --kwargs_dict ${kwargs_dict[@]}
    fi
}

get_running_pids(){
    logs_dir=$1

    # scrape log files and collect all PIDs into an array
    for file in "$logs_dir"/*.log; do
        # use regex to match the number in a string with the syntax [pid=1234]
        # https://unix.stackexchange.com/questions/13466/can-grep-output-only-specified-groupings-that-match
        # ----
        # [TODO] THIS WILL FAIL ON MacOS (invalid option '-P') but I haven't found a more general solution
        # https://stackoverflow.com/questions/77662026/grep-invalid-option-p-error-when-doing-regex-in-bash-script
        # (the '-P' is required for the look-behind, '\K', which excludes "[pid=" from the returned result)
        # ----
        all_pids+=($(grep -oP '\[pid=\K\d+' $file))
    done

    # deduplicate the array
    pids=($(for pid in "${all_pids[@]}"; do echo $pid; done | sort --sort=numeric --unique))
    # add currently running python PIDs
    pids+=($(ps -ef | grep python | awk '{print $2}'))
    # get only values that were in both lists (which are now duplicates in pids)
    running_pids=($(for pid in "${pids[@]}"; do echo $pid; done | sort | uniq --repeated))

    echo ${running_pids[@]}
}

kill_all_pids(){
    run_id=$1
    logs_dir=$2

    kill_pids=($(get_running_pids $logs_dir))

    # killing processes can be dangerous. make the user confirm.
    echo "WARNING, you are about to kill all processes started by the run run_id='${run_id}'."
    echo "This includes at least the following PIDs: ${kill_pids[@]}"
    echo "Enter 'y' to continue or any other key to abort: "
    read continue_kill
    continue_kill="${continue_kill:-n}"

    if [ $continue_kill == "y" ]; then
        echo "Killing."
        # processes may have started or ended while we waited for the user to confirm, so fetch them again
        kill_pids=($(get_running_pids $logs_dir))
        # kill
        for pid in "${kill_pids[@]}"; do kill $pid; done
    else
        echo "Aborting."
    fi
}

monitor_top(){
    nsleep=$1
    logs_dir=$2
    logfile="${logs_dir}/top.txt"
    echo "Monitoring top and saving output. logfile=top.txt"

    running_pids=(1)  # just need a non-zero length array to start
    while [ ${#running_pids[@]} -gt 0 ]; do
        running_pids=($(get_running_pids $logs_dir))
        pid_flags=()
        for pid in ${running_pids[@]}; do pid_flags+=("-p${pid}"); done

        if [ ${#running_pids[@]} -gt 0 ]; then
            {
                echo ----
                date "+%Y/%m/%d %H:%M:%S %Z"
                top -b -n1 -o-PID ${pid_flags[@]}
            } | tee -a $logfile

            sleep $nsleep
        fi
    done
}

print_help(){
    echo "---- $(basename $0) ----"
    echo
    echo "Use this script to launch and monitor a large-scale \"run\" to load a sample of target objects"
    echo "from the literature and collect their multi-wavelength light curves from various archives."
    echo "For complete usage information, see the scale_up notebook tutorial."
    echo
    echo "FLAG OPTIONS"
    echo "------------"
    echo
    echo "Required flags:"
    echo
    echo "    -r 'run_id'"
    echo "        ID for the run. No spaces or special characters."
    echo "        Determines the name of the output directory."
    echo "        Can be used in multiple script calls to manage the same run."
    echo
    echo "Flags used to launch a run (optional):"
    echo
    echo "    -a 'archive names'"
    echo "        Space-separated list of archive names like 'Gaia IceCube WISE' (case insensitive),"
    echo "        or a shortcut ('core' or 'all')."
    echo "        The get_<name>_lightcurves function will be called once for each name."
    echo "        If this flag is not supplied, no light-curve data will be retrieved."
    echo
    echo "    -d 'key=value'"
    echo "        Any top-level key/value pair in the python kwargs_dict where the"
    echo "        value is a basic type (e.g., bool or string, but not list or dict)."
    echo "        Repeat the flag to send multiple kwargs."
    echo "        For more flexibility, use the -j flag and/or store the kwargs_dict as a"
    echo "        yaml file and use: -d 'use_yaml=true'. Order of precedence is dict, json, yaml."
    echo
    echo "    -j 'json string'"
    echo "        The python kwargs_dict as a json string. An example usage is:"
    echo "        -j '{"get_samples": {"SDSS": {"num": 50}}, "archives": {"ZTF": {"nworkers": 8}}}'"
    echo "        The string can be created in python by first constructing the dictionary and then using:"
    echo "            >>> import json"
    echo "            >>> json.dumps(kwargs_dict)"
    echo "        Copy the output, including the surrounding single quotes ('), and paste it as the flag value."
    echo
    echo "Other flags (optional):"
    echo "    These must be used independently and cannot be combined with any other optional flag."
    echo
    echo "    -t 'nsleep'"
    echo "        Use this flag to monitor top after launching a run. This will filter for PIDs"
    echo "        launched by 'run_id' and save the output to a log file once every 'nsleep' interval."
    echo "        'nsleep' will be passed to the sleep command, so values like '10s' and '30m' are allowed."
    echo "        The python helper can load the output."
    echo "        This option is only available on Linux machines."
    echo
    echo "    -k (kill)"
    echo "        Use this flag to kill all processes that were started using the given 'run_id'."
    echo "        This option is only available on Linux machines."
    echo
    echo "    -h (help)"
    echo "        Print this help message."
    echo
}

print_help_invalid_option(){
    echo "For help, use -h."
}

print_logs(){
    logfile=$1
    echo "-- ${logfile}:"
    cat $logfile
    echo "--"
}

# ---- Set variable defaults.
archive_names=()  # "core", "all", or space-separated list of names like "gaia wise"
kwargs_dict=()
kwargs_json='{}'
kill_all_processes=false

# ---- Set variables that were passed in as script arguments.
while getopts r:a:d:j:t:kh flag; do
    case $flag in
        r) run_id=$OPTARG
            kwargs_dict+=("run_id=${OPTARG}")
            ;;
        a) archive_names=("$OPTARG");;
        d) kwargs_dict+=("$OPTARG");;
        j) kwargs_json=$OPTARG;;
        t) nsleep=$OPTARG;;
        k) kill_all_processes=true;;
        h) print_help
            exit 0
            ;;
        ?) print_help_invalid_option
            exit 1
            ;;
      esac
done

# If a run_id was not supplied, exit.
if [ -z $run_id ]; then
    echo "$(basename $0): missing required option -- 'r'"
    print_help_invalid_option
    exit 1
fi

# ---- Request some kwarg values from HELPER_PY.
base_dir=$(RUN_HELPER_PY base_dir+)
# if HELPER_PY didn't create base_dir then something is wrong and we need to exit
if [ ! -d "$base_dir" ]; then
    echo "${base_dir} does not exist. Exiting."
    exit 1
fi
parquet_dir=$(RUN_HELPER_PY parquet_dir+)
logs_dir=$(RUN_HELPER_PY logs_dir+)
# expand an archive_names shortcut value.
if [ "${archive_names[0]}" == "all" ]; then archive_names=($(RUN_HELPER_PY archive_names_all+l)); fi
if [ "${archive_names[0]}" == "core" ]; then archive_names=($(RUN_HELPER_PY archive_names_core+l)); fi

# ---- Construct logs paths.
mkdir -p $logs_dir
mylogfile="${logs_dir}/$(basename $0).log"

# ---- If the user has requested to monitor with top, do it and then exit.
if [ $nsleep ]; then
    monitor_top $nsleep $logs_dir
    exit 0
fi

{  # we will tee the output of everything below here to $mylogfile

# ---- If the user has requested to kill processes, do it and then exit.
if [ $kill_all_processes == true ]; then
    kill_all_pids $run_id $logs_dir $mylogfile
    exit 0
fi

# ---- Report basic info about the run.
echo "*********************************************************************"
echo "**                          Run starting.                          **"
echo "run_id=${run_id}"
echo "base_dir=${base_dir}"
echo "logs_dir=${logs_dir}"
echo "parquet_dir=${parquet_dir}"
echo "**                                                                 **"

# ---- Do the run. ---- #

# ---- 1: Run job to get the object sample, if needed. Wait for it to finish.
logfile_name="get_sample.log"
logfile="${logs_dir}/${logfile_name}"
echo
echo "Build sample is starting. logfile=${logfile_name}"
RUN_HELPER_PY sample >> ${logfile} 2>&1
echo "Build sample is done. Printing the log for convenience:"
echo
print_logs $logfile

# ---- 2: Start the jobs to fetch the light curves in the background. Do not wait for them to finish.
echo
echo "Archive calls are starting."
echo
for archive in ${archive_names[@]}; do
    logfile_name="$(awk '{ print tolower($0) }' <<< $archive).log"
    logfile="${logs_dir}/${logfile_name}"
    RUN_HELPER_PY lightcurves $archive >> ${logfile} 2>&1
    echo "[pid=${!}] ${archive} started. logfile=${logfile_name}"
done

# ---- 3: Print some instructions for the user, then exit.
echo
echo "**                                                                  **"
echo "**                       Main process exiting.                      **"
echo "**           Jobs may continue running in the background.           **"
echo "**********************************************************************"
} | tee -a $mylogfile
