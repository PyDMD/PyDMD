import json

config = {
    "testing_strategy_ghact": {
        "fail-faste": "false",
        "python-version":  
            [
                "3.8",    
                "3.9"
            ],
        "os":
            [
                "windows-latest",                                 
                "macos-latest",                 
                "ubuntu-latest" 
            ]    
    }       
}   

print(json.dumps(config))
