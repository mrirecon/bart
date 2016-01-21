# bart parameter-completion



function _bart()
{
	local cur=${COMP_WORDS[$COMP_CWORD]}

	if [ $COMP_CWORD -eq 1 ] ; then

		local CMDS=$(bart | tail -n +2)
		COMPREPLY=($(compgen -W "$CMDS" -- "$cur"));

	else

		local bcmd=${COMP_WORDS[1]}

		case $cur in
		-*)
			COMPREPLY=($(bart ${bcmd} -h | grep -o -E "^${cur}\w*"))
			;;
		*)
			case $bcmd in
			twixread)
				COMPREPLY=($(compgen -o plusdirs -f -X '!*.dat' -- ${cur}))
				;;
			*)
				local CFLS=$(compgen -o plusdirs -f -X '!*.hdr' -- ${cur})
				local COOS=$(compgen -o plusdirs -f -X '!*.coo' -- ${cur});
				local RAS=$(compgen -o plusdirs -f -X '!*.ra' -- ${cur});
				local suffix=".hdr"
				COMPREPLY=($(for i in ${CFLS} ${COOS} ${RAS}; do echo ${i%$suffix} ; done))
				;;
			esac

			;;
		esac
	fi

	return 0
}

complete -o filenames -F _bart bart ./bart

