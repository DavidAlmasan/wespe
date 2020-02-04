# deep-learning-marchantia

### Please see [SCRIPTS](./SCRIPTS) folder for information on how to train and run models.

### The final report can be found [here](./final_report_v3.pdf)

Information on remote connections:
- editing using VSCode (need rmate and Remote VSCode extension)
  - follow this tutorial: https://medium.com/@prtdomingo/editing-files-in-your-linux-virtual-machine-made-a-lot-easier-with-remote-vscode-6bb98d0639a4
  - ssh -R 52698:localhost:52698
  - in the future will make a shortcut for the tunnel element
- enabling jupyter notebooks on remote machine
  - use SHH tunnelling (twice!)
  - ssh -L<port_A>:localhost:<port_B> user@Host-B (use port 8888)
  - see tutorials: https://medium.com/@sankarshan7/how-to-run-jupyter-notebook-in-server-which-is-at-multi-hop-distance-a02bc8e78314

Changing Tensorflow versions from 1.11.0 to 1.9.0:
- CuDNN version on hacktar is 7.1.3, and the loaded library was 7.0.3
  - The pip package for tensorflow 1.11.0 was compiled with CuDNN 7.2.1, so the versions were not compatible
- Installing tensorflow 1.9.0 allowed the CuDNN versions to match (without upgrading CuDNN or installing tensorflow from source, both of which requires sudo rights)
