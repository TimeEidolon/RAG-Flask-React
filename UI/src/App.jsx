import React, {Suspense} from 'react';
import {BrowserRouter as Router} from 'react-router-dom';
import {AppRoutes} from './config/routes';
import ChatBall from "./components/Chat/ChatBall.jsx";

function App() {
    return (
        <>
            <ChatBall/>

            <Suspense fallback={<div>Loading...</div>}>
                <Router>
                    <AppRoutes/>
                </Router>
            </Suspense>
        </>
    );
}


export default App
