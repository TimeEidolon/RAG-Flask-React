import {io} from 'socket.io-client';

const socket = io('/tasks');

useEffect(() => {
    socket.on('task_update', (data) => {
        if (data.task_id === currentTaskId) {
            setTaskStatus(data.status);
        }
    });
    return () => socket.disconnect();
}, []);