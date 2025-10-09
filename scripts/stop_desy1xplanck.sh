### unlink emulator likelihoods from the usual cocoa likelihood
if [ -d "$ROOTDIR/projects/desy1xplanck/emulator_deprecated/likelihood" ]; then
	rm $ROOTDIR/projects/desy1xplanck/likelihood/*_emu.py
    rm $ROOTDIR/projects/desy1xplanck/likelihood/*_emu.yaml
fi


unset SPDLOG_LEVEL
