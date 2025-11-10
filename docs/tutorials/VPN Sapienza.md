# VPN Sapienza & Cluster

## Connettersi alla VPN

Per connettersi alla **VPN** utilizzare il seguente comando.

```
snx -u {nome.cognome}@uniroma1.it -s castore.uniroma1.it
```

Successivamente inserire la *password* associata all'accaunt istituzionale.

Per disconnettersi usare il seguente comando.

```
snx -d
```





## Connessione

### Account 

Occorre un account **Cerbero** da richiedere agli amministratori.

### Cluster

Per prima cosa bisogna connettersi al **frontend** con *ssh*

```
ssh {user}@151.100.174.45
```

Successivamente inserire la *password* associata.  Ora ci si può connettere al **submitter**.

```
ssh submitter
```

Occorrerà di nuovo inserire la *password* associata, la stessa utilizzata per il **frontend**.


### Connessione automatica

Aggiungendo al file `~/.ssh/config` le seguenti righe:

```
Host cluster.myfrontend.vpn
    HostName 151.100.174.45
    Port 22
    User {user}
    
Host cluster.submit.vpn
    HostName 192.168.0.102
    ProxyJump cluster.myfrontend.vpn
    Port 22
    User {user}
```

Sarà possibile connettersi al cluster direttamente con il comando sotto:

```
ssh cluster.submit.vpn
```

Sarà sempre necessario fornire la *password* per l'account nel cluster.





## Singularity

Sul cluster i container sono gestiti da **Singularity**.  Il comando per generare un immagine di Singularity (**SIF**) è il seguente:
```
[sudo] singularity build [--sandbox] {image_name} {target}
```

Per `target` si intende l'immagine **Docker** da cui generare l'immagine `.sif` oppure il *Definition File*.  Senza l'opzione `--sandbox` l'immagine è **read-only** e il formato sarà `.simg`.

È quindi possibile creare un'immagine *Singularity* partendo da un'immagine *Docker* locale con il comando:

```
[sudo] singularity build [--sandbox] {singularity_image_name} docker-daemon://{docker_image_name}
```




## Slurm

Per eseguire i *job* bisogna utilizzare **Slurm**. Con `sinfo` e `squeue` si può monitorare lo stato dei *job* (o del singolo *job* d'interesse).

Per lanciare un *job* occorre usare il comando:

```
srun -p {partition} {command}
```

Per specificare requisiti per il *job*, flag comuni sono:
- `--gpus`: numero minimo di GPU che il nodo deve avere per eseguire il *job*;
- `--mem`: la memoria in MB richiesta per il nodo;





## Condivisione Memoria

Per **caricare** file **SUL** *cluster*:

```
rsync -vr {file} {user}@151.100.174.45:~/{user}/{remote_directory}
```

Per **scaricare** file **DAL** *cluster*:

```
rsync -vr {user}@151.100.174.45:~/{user}/{file} {local_directory}
```


