import {Routes, Route, Navigate} from 'react-router-dom';
import React, {lazy} from 'react';

const HomePage = lazy(() => import('../components/HomePage'));
const FinancialReportForIndex = lazy(() => import('../components/FinancialReport/FinancialReportForIndex'));

export const AppRoutes = () => (
    <Routes>
        <Route path="/" element={<Navigate to="/home" replace />} />
        <Route path="/home" element={<HomePage/>}/>
        <Route path="/financial_report" element={<FinancialReportForIndex/>}/>
    </Routes>
);