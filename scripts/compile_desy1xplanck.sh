# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
if [ -z "${IGNORE_COSMOLIKE_DESY1XPLANCK_CODE}" ]; then

  if [ -z "${ROOTDIR}" ]; then
    source start_cocoa.sh || { pfail 'ROOTDIR'; return 1; }
  fi

  # Parenthesis = run in a subshell
  ( source "${ROOTDIR:?}/installation_scripts/flags_check.sh" ) || return 1;
    
  unset_env_vars () {
    unset -v PROJECT FOLDER
    cdroot || return 1;
  }

  unset_env_funcs () {
    unset -f cdfolder cpfolder cpfile error
    unset -f unset_env_funcs
    cdroot || return 1;
  }

  unset_all () {
    unset_env_vars
    unset_env_funcs
    unset -f unset_all
    cdroot || return 1;
  }

  error () {
    fail_script_msg "$(basename "${BASH_SOURCE[0]}")" "${1}"
    unset_all || return 1
  }
    
  cdfolder() {
    cd "${1:?}" 2>"/dev/null" || { error "CD FOLDER: ${1}"; return 1; }
  }

  cpfolder() {
    cp -r "${1:?}" "${2:?}"  \
      2>"/dev/null" || { error "CP FOLDER ${1} on ${2}"; return 1; }
  }
  
  cpfile() {
  cp "${1:?}" "${2:?}" \
    2>"/dev/null" || { error "CP FILE ${1} on ${2}"; return 1; }
  }

  # ---------------------------------------------------------------------------
  # ---------------------------------------------------------------------------
  # ---------------------------------------------------------------------------

  unset_env_vars || return 1

  # ---------------------------------------------------------------------------

  PROJECT="${ROOTDIR:?}/projects"

  FOLDER="${DESXPLANCK_GIT_NAME:-"desy1xplanck"}"

  PACKDIR="${PROJECT:?}/${FOLDER:?}"

  # Name to be printed on this shell script messages
  PRINTNAME="DESY1xPLANCK"

  ptop "COMPILING ${PRINTNAME:?}" || return 1

  # ---------------------------------------------------------------------------
  # cleaning any previous compilation
  rm -rf "${PACKDIR:?}"/interface/*.o
  rm -rf "${PACKDIR:?}"/interface/*.so
  cd "${PACKDIR}"/interface
  make -f MakefileCosmolike clean >${OUT1:?} 2>${OUT2:?} || { error "${EC2:?}"; return 1; }

  # ---------------------------------------------------------------------------
  cd "${PACKDIR}"/interface

  (export LD_LIBRARY_PATH=${CONDA_PREFIX:?}/lib:$LD_LIBRARY_PATH && \
   export LD_LIBRARY_PATH=${ROOTDIR:?}/.local/lib:$LD_LIBRARY_PATH && \
   make -j $MNT -f MakefileCosmolike all >${OUT1:?} 2>${OUT2:?} || { error "${EC8:?}"; return 1; })

  cd ${ROOTDIR:?}

  pbottom "COMPILING ${PRINTNAME:?}" || return 1

  # ---------------------------------------------------------------------------

  unset_all || return 1

fi

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
