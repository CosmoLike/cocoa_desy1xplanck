
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
if [ -z "${IGNORE_COSMOLIKE_DESY1XPLANCK_CODE}" ]; then

  if [ -z "${ROOTDIR}" ]; then
      echo 'ERROR ROOTDIR not define'
      return
  fi

  # Parenthesis = run in a subshell
  ( source "${ROOTDIR:?}/installation_scripts/flags_check.sh" ) || return 1;

  export LD_LIBRARY_PATH="${ROOTDIR:?}/projects/desy1xplanck/interface":${LD_LIBRARY_PATH}

  export PYTHONPATH="${ROOTDIR:?}/projects/desy1xplanck/interface":${PYTHONPATH}

  if [ -n "${COSMOLIKE_DEBUG_MODE}" ]; then
      export SPDLOG_LEVEL=debug
  else
      export SPDLOG_LEVEL=info
  fi
fi
