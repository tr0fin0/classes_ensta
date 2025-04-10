# CSC_5RO05_TA

In this class a remote Raspberry Pi 2B will be used for development. This Pi is connected to ENSTA's servers, therefore 2 SSHs are necessary.

## VSCode Configuration

Modify VSCode configurations and add the following:

```bash
Host ensta-ssh
    HostName ssh.ensta.fr
    User trofino

Host rpi2b-dev
    HostName 147.250.8.198
    User g.trofino

Host rpi2b-dev-remote
    HostName 147.250.8.198
    User g.trofino
    ProxyJump ensta-ssh
```

## Connection

### `rpi2b-dev`

Is used to connect via internal ENSTA's network.

### `rpi2b-dev-remote`

Is used to connect via external ENSTA's network. It is necessary to use CASCAD login in configuration and password in connection.

### Raspberry Pi

`.raspberry_number` file contains the number of the Pi used for the user.

To connect to the Pi:

```bash
ssh root@192.168.50.44
```

Type `yes` then enter the password: `geXDsIC9`

## Development

Algorithms may be developed in the remote environment as it already contains the compilers necessary.

### Compilation

Compilation needs to be done with ARM compiler to suit Raspberry Pi requirements.

```bash
arm-linux-g++ -Wall -Wextra main.cpp lib.cpp -o main
```
Then copy the binary to the Pi with:

```bash
rsync -avz main root@192.168.50.44:
```

Finally execute the algorithm on the Pi with:

```bash
./main
```

## Reference

[CSC_5RO05_TA](https://irfu.cea.fr/Pisp/shebli.anvar/prog-mt-rtos/md__h_1_2_activites_2_enseignement_2_cours-2025-_web_2ensta_2_e_n_s_t_a-_instructions.html)
