#!/bin/bash

# Empyrean setup file for "bash"
#
# NOTE: INSTALL_PATH is the name of installation;
#       LICENSE_HOST is the license server's hostname (or IP address);
#       LICENSE_PORT is a port number between 1024 and 65535 
#
if [ $?LM_LICENSE_FILE ]; then
  export LM_LICENSE_FILE=59001@127.0.0.1:$LM_LICENSE_FILE
else
  export LM_LICENSE_FILE=59001@127.0.0.1
fi

export ALPS_ROOT=/home/toolbox/ALPS
export PATH=$ALPS_ROOT/bin:$PATH
export ALPS_HOME=$ALPS_ROOT/tools/alps

export ALPSCD_HOME=$ALPS_ROOT/tools/alpscd
