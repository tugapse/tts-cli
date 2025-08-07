# Bash Autocompletion for the 'tts-cli' command
# This script handles all flags and arguments defined in the argparse setup.

_tts-cli_completion() {
    local cur prev words cword
    _init_completion || return

    COMPREPLY=()
    local all_flags_and_options=""

    # List all main flags and options
    all_flags_and_options+="--build "
    all_flags_and_options+="--auto-accept -y "
    all_flags_and_options+="--model-type "
    all_flags_and_options+="--text "
    all_flags_and_options+="--output-file -o "
    all_flags_and_options+="--model-name "
    all_flags_and_options+="--language "
    all_flags_and_options+="--speaker-embedding_path "
    all_flags_and_options+="--device "
    all_flags_and_options+="--temperature "
    all_flags_and_options+="--top_k "
    all_flags_and_options+="--top_p "
    all_flags_and_options+="--max_new_tokens "
    all_flags_and_options+="--description-prompt -dp "
    all_flags_and_options+="--voice -vc "
    all_flags_and_options+="--debug-console -dc "
    all_flags_and_options+="--quiet -q "
    all_flags_and_options+="--help -h " # Always include help

    # If the current word starts with a dash, suggest flags
    if [[ "${cur}" == -* ]]; then
        COMPREPLY=( $(compgen -W "${all_flags_and_options}" -- "${cur}") )
        return
    fi
    
    # Handle specific argument completions based on the previous flag
    case "${prev}" in
        --model-type)
            # Suggest choices for --model-type
            local model_types="parler orpheus"
            COMPREPLY=( $(compgen -W "${model_types}" -- "${cur}") )
            return
            ;;
        --voice|-vc)
            # Suggest choices for --voice (Orpheus model voices)
            local voices="tara leah jess leo dan mia zac zoe"
            COMPREPLY=( $(compgen -W "${voices}" -- "${cur}") )
            return
            ;;
        --output-file|-o|--speaker-embedding_path)
            # Suggest files for output and speaker embedding paths
            _filedir
            return
            ;;
        # For --text, --model-name, --language, --description-prompt, etc.,
        # the completion is typically free-form text, so no specific `compgen`
        # rule is needed beyond the default shell behavior.
    esac

    # If no specific flag completion, and it's the first positional argument,
    # it's likely the 'input_text' argument. We can offer basic filename completion
    # or just let the user type freely. For simplicity, we'll offer filename completion.
    if [[ "${cword}" -eq 1 ]]; then
        _filedir
        return
    fi

    # Fallback to default filename completion if no specific rule applies
    _filedir
}

# Register the completion function for the 'tts-cli' command
complete -F _tts-cli_completion tts-cli