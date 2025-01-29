$prompts = @(
    "Create a short story about a robot discovering nature for the first time. Generate 1000 words.",
    "What is the first letter of english alphabet?",
    '''Generate a 5000 word Comprehensive Analysis of Global Climate Change Impact on Agriculture
    Context: The global agricultural sector is facing unprecedented challenges due to climate change. Rising temperatures, 
    shifting precipitation patterns, and increasing frequency of extreme weather events are threatening food security and 
    livelihoods worldwide. This analysis will explore the multifaceted impact of climate change on agriculture, drawing on 
    extensive data from the past two decades, scientific research, and case studies from different regions.
    Sections:
    Introduction:
    Overview of climate change and its significance.
    Brief history of agricultural development and dependence on stable climate conditions.
    Purpose and scope of the analysis.
    Effects on Crop Yield:
    Historical data on crop yields and climate variables.
    Impact of rising temperatures on crop growth cycles, photosynthesis, and productivity.
    Analysis of changes in precipitation patterns and their effects on soil moisture and irrigation needs.
    Case studies of specific crops (e.g., wheat, rice, corn) and regions (e.g., Sub-Saharan Africa, Southeast Asia).
    Regional Impacts:
    Detailed examination of how climate change affects agriculture in different regions.
    Focus on vulnerable regions and those experiencing significant changes.
    Socioeconomic implications for farmers and rural communities.
    Adaptation measures and their effectiveness in various contexts.
    Adaptation Strategies:
    Overview of current and potential adaptation strategies to mitigate climate impacts.
    Technological innovations in agriculture (e.g., drought-resistant crops, precision farming).
    Policy and governance approaches to support adaptation (e.g., subsidies, insurance schemes).
    Role of international cooperation and funding in facilitating adaptation.
    Conclusion:
    Summary of key findings and insights.
    Recommendations for policymakers, agricultural stakeholders, and researchers.
    Future research directions and potential developments in the field.
    References:
    Cite a comprehensive list of scientific papers, reports, and case studies used in the analysis.
    '''
    # "What is 1+1",
    # "Generate a recipe for a vegan chocolate cake.",
    # "What is your name",
    # "Write a poem about the changing seasons.",
    # "Output this word: foo",
    # '''In-Depth Review of "Sapiens: A Brief History of Humankind" by Yuval Noah Harari
    # Context: Sapiens: A Brief History of Humankind by Yuval Noah Harari offers a sweeping narrative of human history, 
    # exploring how Homo sapiens came to dominate the Earth. This review will provide an in-depth examination of Hararis 
    # main arguments, a detailed summary of each section, and a critical analysis comparing his perspectives with 
    # other historians. The goal is to engage deeply with the text and explore its impact on contemporary understanding of human history.
    # Sections:
    # Introduction:
    # Background information on Yuval Noah Harari and his academic credentials.
    # Overview of the books publication history and reception.
    # Purpose and scope of the review.
    # Main Arguments:
    # Detailed exposition of Hararis primary thesis and key arguments.
    # Discussion of the cognitive revolution, agricultural revolution, unification of humankind, and scientific revolution.
    # Exploration of Hararis views on happiness, power dynamics, and future trajectories of Homo sapiens.
    # Summary of Each Section:
    # Part 1: The Cognitive Revolution
    # Overview of early human history and the development of cognitive abilities.
    # Impact of language, imagination, and social cooperation on human evolution.
    # Part 2: The Agricultural Revolution
    # Transition from hunter-gatherer societies to agricultural communities.
    # Effects of agriculture on social structures, economies, and human health.
    # Part 3: The Unification of Humankind
    # Examination of the forces that unified human societies, including empires, religions, and trade.
    # Analysis of cultural exchange and conflict.
    # Part 4: The Scientific Revolution
    # Exploration of the rise of modern science and its transformative effects on human societies.
    # Impact of technological advancements and industrialization.
    # Critical Analysis:
    # Strengths of Hararis arguments and narrative style.
    # Critical examination of potential biases and areas of controversy.
    # Comparison with perspectives from other historians and scholars.
    # Discussion of the books influence on popular and academic discourse.
    # Conclusion:
    # Summary of the reviews key points.
    # Personal reflections on the books impact and significance.
    # Recommendations for further reading and study.
    # References:
    # Comprehensive list of sources, including books, articles, and academic papers referenced in the review.''',
    # "What is the capital of France?",
    # "Summarize the plot of Pride and Prejudice by Jane Austen in 100 words.",
    # "What is the square root of 144?",
    # "Describe the process of photosynthesis.",
    # "What is the largest planet in our solar system?",
    # "Generate a list of 10 programming languages and their primary uses.",
    # "Write a 20 word biography of Albert Einstein.",
    # "Write a dialogue between two friends debating the pros and cons of space exploration.",
    # "What is the boiling point of water in Fahrenheit?",
    # "Create an itinerary for a 3-day trip to Tokyo, including places to visit and food to try.",
    # "What is the chemical symbol for gold?",
    # "Explain how blockchain technology works.",
    # "What is the population of New York City?",
    # "Write a brief history of the Internet.",
    # "What is the speed of light in a vacuum?",
    # "Provide 5 tips for improving time management skills.",
    # "What is the difference between a simile and a metaphor?",
    # "Generate a fictional conversation between Leonardo da Vinci and Nikola Tesla.",
    # "What is the atomic number of carbon?",
    # "Describe the cultural significance of the cherry blossom festival in Japan.",
    # "What is the meaning of the word 'serendipity'?",
    # "Summarize the main events of World War II.",
    # "What is the capital of Australia?",
    # "Write a persuasive essay on the importance of renewable energy.",
    # "What is the chemical formula for water?",
    # "Create a character profile for a detective in a mystery novel.",
    # "What is the population of China?",
    # "Generate a list of 10 must-read books for science fiction fans.",
    # "List three breakfast foods that are high in protein.",
    # "Explain the impact of social media on modern communication."
)

#loop through promots for the specified test time at each power limit
$testTime = 240
$testCount = 0
$limiting_mode = "none" #could also be frequency, agent, or none
$print_responses = $false
$debug = $false
#$model_list = @("llama3.1:8b", "llama3.2", "llama3.3")
$model_list = @("llama3.3")
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
$process = Start-Process -NoNewWindow -FilePath "/root/miniconda3/envs/gpu_load_line/bin/python" -ArgumentList "monitor_nvidia.py" -PassThru

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
    $testTime = 20
}
$variationCount = 0
$testCount = 0
#get the gpu model from nvidia-smi
#$gpu_model = (nvidia-smi --query-gpu=name --format=csv | Select-String -Pattern "^[a-zA-Z0-9 ]+$" | ForEach-Object { $_ -replace '\s+', '' } | Select-Object -First 1)
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
    elseif($limiting_mode -eq "none" -or $limiting_mode -eq "agent") {
        if ($testCount -ge $prompts.Length) {
            break
        }
    }
    else {
        #do nothing as this is the base case 
        Write-Output "Unknown power limiting mode set: $limiting_mode"
    }

    $startTime = Get-Date
    $endTime = $startTime.AddSeconds($testTime)
    Start-Sleep -Seconds 1
    for ($i = 0; $i -lt $prompts.Length; $i++) {

        # if (($limiting_mode -eq "frequency" -or $limiting_mode -eq "power") -and $endTime -lt (Get-Date)) {
        #     Write-Output "Test time reached. Breaking"
        #     break inner
        # }

        $body = @{
            #pick the model based on modulo of the prompt index
            model = $model_list[$i % $model_list.Length]
            #model = "llama3.2"
            #model = "llama3.1:8b"
            #model = "llama3.3"
            prompt = $prompts[$i]
        }
        #print the prompt
        Write-Output "$i of $($prompts.Length) Prompt: $($prompts[$i])"

        $queryStartTime = Get-Date -Format "yyyy-MM-dd HH:mm:ss.fff"
        $jsonResponse = Invoke-RestMethod -Uri "http://localhost:11434/api/generate" -Method Post -Body ($body | ConvertTo-Json) -ContentType "application/json"
        $queryEndTime = Get-Date -Format "yyyy-MM-dd HH:mm:ss.fff"

        # Split the concatenated JSON string into individual JSON objects
        $jsonObjects = $jsonResponse -split "`n"

        # Initialize an array to hold the PowerShell objects
        $responseArray = @()

        # Convert each JSON object string into a PowerShell object and add it to the array
        foreach ($jsonObject in $jsonObjects) {
            $responseArray += $jsonObject | ConvertFrom-Json
        }

        if ($print_responses) {
            #print the response field of each element except last in the array
            for ($j = 0; $j -lt $responseArray.Length - 1; $j++) {
                Write-Host "$($responseArray[$j].response)" -NoNewline
            }
        }
        # Find the response where "done": true
        $finalResponse = $responseArray | Where-Object { $_.done -eq $true }

        # Calculate the total number of tokens
        $totalTokens = $finalResponse.context.Count

        # Calculate the total duration in seconds
        $totalDurationSeconds = $finalResponse.total_duration / 1e9  # Convert nanoseconds to seconds

        Write-Output ""

        # Calculate tokens per second
        if ($totalDurationSeconds -gt 0) {
            $tokensPerSecond = $totalTokens / $totalDurationSeconds
            Write-Output "Total Tokens: $totalTokens"
            Write-Output "Total Duration (seconds): $totalDurationSeconds"
            Write-Output "Tokens per Second: $tokensPerSecond"

            #output to csv the start, end and the token stats
            $csvOutput = [PSCustomObject]@{
                "StartTime" = $queryStartTime
                "EndTime" = $queryEndTime
                "TotalTokens" = [float]$totalTokens
                "TotalDuration(seconds)" = [float]$totalDurationSeconds
                "TokensPerSecond" = [float]$tokensPerSecond
                #if prompt id is odd then long else short
                "LongOrShortPrompt" = if ($i % 2 -eq 0) { "Long" } else { "Short" }
                "Model" = $body.model
            }
            $csvOutput | Export-Csv -Append -Path "inference_load.csv" -NoTypeInformation
        } else {
            Write-Output "No valid duration found."
        }
        #use this if doing agent test 
        $testCount++
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
