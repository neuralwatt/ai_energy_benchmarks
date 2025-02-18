

#loop through promots for the specified test time at each power limit
$testCount = 0
$limiting_mode = "none" #could also be frequency, agent, power, or none
$debug = $false
$model_list = @("llama2")
#setup
#before testing make sure graphics clocks are in default mode
nvidia-smi -rgc

if ($limiting_mode -eq "power") {
    # Get the min and max nvidia-smi power limits
    $min_power_limit = (nvidia-smi -q -d POWER | Select-String "Min Power Limit" | ForEach-Object { $_ -replace '\D', '' } | Select-Object -First 1)
    $max_power_limit = (nvidia-smi -q -d POWER | Select-String "Max Power Limit" | ForEach-Object { $_ -replace '\D', '' } | Select-Object -First 1)

    # Convert the extracted values to integers
    # When parsed these dropped the decimal so divide by 100
    $min_power_limit = [int]$min_power_limit/100
    $max_power_limit = [int]$max_power_limit/100

    # Create a list of power limits by 10% increments from min to max
    $power_limits = @()
    for ($i = $min_power_limit; $i -le $max_power_limit; $i += [math]::Ceiling(($max_power_limit - $min_power_limit) * 0.1)) {
        $power_limits += $i
    }
    
    #remove the last power limit and replace with the max power limit
    $power_limits = $power_limits[0..($power_limits.Length - 2)]

    #last power limit should just be the max
    $power_limits += $max_power_limit

    # Print the power limits
    Write-Output "Power limits: $power_limits"
}
elseif ($limiting_mode -eq "frequency")
{
    #get the set of valid gpu frequencies
    $gpu_frequencies_output = nvidia-smi --query-supported-clocks=graphics,memory --format=csv
    $valid_gpu_frequencies = $gpu_frequencies_output | Select-String -Pattern "^[0-9]+ MHz" | ForEach-Object { ($_ -split ",")[0].Trim() -replace ' MHz', '' }
    $valid_gpu_frequencies = $valid_gpu_frequencies | ForEach-Object { [int]$_ }
    #sort the frequencies and remove duplicates
    $valid_gpu_frequencies = $valid_gpu_frequencies | Sort-Object -Unique

    # create a list of frequencies by 10% increments from min to max
    # only choose valid frquencies so use the index to the list for calc
    $gpu_frequencies = @()
    for ($i = 0; $i -lt $valid_gpu_frequencies.Length; $i += [math]::Ceiling($valid_gpu_frequencies.Length * 0.1)) {
        $gpu_frequencies += $valid_gpu_frequencies[$i]
    }

    #remove last frequency and replace with teh last element of the list
    $gpu_frequencies = $gpu_frequencies[0..($gpu_frequencies.Length - 2)]
    $gpu_frequencies += $valid_gpu_frequencies[-1]

    Write-Output "GPU Frequencies: $gpu_frequencies"
}
elseif ($limiting_mode -eq "agent") {
    #launch the co2cpufrequency in gpu-energy-mode
    $co2process = Start-Process -NoNewWindow -FilePath "/home/jovyan/work/CarbonAwareLinux/os/carbon_aware_governor/target/debug/co2cpufrequency" -ArgumentList "--gpu-energy-mode --update-interval-sec=1" -PassThru
}
else {
    Write-Output "Power limiting mode set: $limiting_mode; no initialization action for this mode"
}

#start monitor_nvidia.py
$process = Start-Process -NoNewWindow -FilePath "python" -ArgumentList "monitor_nvidia.py" -PassThru

# Get the PID of the process
$monitor_id = $process.Id 

#if debug then limit to last two power limits
if ($debug) {
    if ($limiting_mode -eq "power") {
        $power_limits = $power_limits[-2..-1]
    }
    elseif ($limiting_mode -eq "frequency") {
        $gpu_frequencies = $gpu_frequencies[-2..-1]
    }
}
$variationCount = 0
$testCount = 0
while ($true) {
    if ($limiting_mode -eq "power") {
        if ($variationCount -ge $power_limits.Length) {
            break
        }
        #set the power limit to the next value in the list
        $power_limit = $power_limits[$variationCount]
        Write-Output "Setting power limit to $power_limit"
        nvidia-smi -pl $power_limit
    }
    elseif ($limiting_mode -eq "frequency") {
        if ($variationCount -ge $gpu_frequencies.Length) {
            Write-Output "Test count: $variationCount, breaking"
            break
        }
        #set the gpu frequency to the next value in the list
        $gpu_frequency = $gpu_frequencies[$variationCount]
        Write-Output "Setting GPU frequency to $gpu_frequency"
        #lgc locks gpu clocks to the range specified
        nvidia-smi -lgc "$gpu_frequency,$gpu_frequency"
    }
    elseif ($limiting_mode -eq "none" -or $limiting_mode -eq "agent") {
        if ($testCount -ge 1) {
            break
        }
    }
    else {
        #do nothing as this is the base case 
        Write-Output "Unknown power limiting mode set: $limiting_mode"
    }

    #define combinations of batch_size and max_seq_len which always add up to 4096
    $batch_size_list = @("8", "16", "32", "64", "128", "256", "512", "1024", "2048")
    #$max_seq_len_list = @("2048", "1024", "512", "256", "128", "64", "32", "16", "8")
    if ($limiting_mode -eq "none" -or $limiting_mode -eq "agent") {
        $batch_size_list = @("16")
    }
    $startTime = Get-Date
    Start-Sleep -Seconds 1
    for ($i = 0; $i -lt $model_list.Length; $i++) {
        for ($b = 0; $b -lt $batch_size_list.Length; $b++ ) {
            $model = $model_list[$i]
            $batch_size = $batch_size_list[$b]
            $max_seq_len = 4096 
            
            $steps = 9033 / 4 / $batch_size
            #kick off the training
            $training_process = Start-Process -NoNewWindow -FilePath "/root/miniconda3/envs/gpu_load_line/bin/tune" -ArgumentList "run lora_finetune_single_device --config finetuning_llama2_config.yaml epochs=1 tokenizer.max_seq_len=$max_seq_len batch_size=$batch_size" -PassThru
            
            #wait until process exists
            $training_process.WaitForExit()
            $endTime = Get-Date
            #output to csv the start, end and the token stats
            $totalDurationSeconds = ($endTime - $startTime).TotalSeconds
            $csvOutput = [PSCustomObject]@{
                "StartTime" = $startTime
                "EndTime" = $endTime
                "TotalDuration(seconds)" = [float]$totalDurationSeconds
                "BatchSize" = $batch_size
                "MaxSeqLen" = $max_seq_len
                "MaxSteps" = $steps
                "Model" = $model
            }
            $csvOutput | Export-Csv -Append -Path "training_load.csv" -NoTypeInformation
            #use this if doing agent test 
            $testCount++
        } 
    }
    #use this if generating the load line
    $variationCount++
}

#reset gpu clocks to default
nvidia-smi -rgc

# Kill the process using the PID
Stop-Process -Id $monitor_id -Force
if ($limiting_mode -eq "agent") {
    Stop-Process -Id $co2process.Id -Force
}
