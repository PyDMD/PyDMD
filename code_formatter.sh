#!/bin/bash

#######################################

required_command="yapf unexpand"
code_directories="dmd tests"

#######################################

usage() {
	echo
	echo -e "\tUsage: $0 [files]"
	echo
	echo -e "\tIf not files are specified, script formats all ".py" files"
	echo -e "\tin code directories ($code_directories); otherwise, formats"
	echo -e "\tall given files"
	echo
	echo -e "\tRequired command: $required_command"
	echo
	exit 0
}


[[ $1 == "-h" ]] && usage

# Test for required program
for comm in $required_command; do
	command -v $comm >/dev/null 2>&1 || {
		echo "I require $comm but it's not installed. Aborting." >&2; 
		exit 1
	}
done

# Find all python files in code directories
python_files=""
for dir in $code_directories; do
	python_files="$python_files `ls $dir/*.py`"
done
[[ $# != 0 ]] && python_files=$@


# Here the important part:
# - first, yapf format the files; it works very well just setting few option
#		by command line, but at the moment it has bugs for tabs, so uses spaces.
# - second, convert 4 spaces to tab character
# - third, you can look a very pretty code 
for file in $python_files; do
	echo "Making beatiful $file..."
	[[ ! -f $file ]] && echo "$file does not exist; $0 -h for more info" && exit
	
	yapf --style='{
					based_on_style: pep8, 
				   	indent_width: 4,
					dedent_closing_brackets: true,
					column_limit: 80
			  	  }' -i $file
	unexpand -t 4 $file > tmp.py
	mv tmp.py $file
done
