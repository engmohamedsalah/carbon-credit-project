import { configureStore } from '@reduxjs/toolkit';
import authReducer from './authSlice';
import projectReducer from './projectSlice';
import verificationReducer from './verificationSlice';

const store = configureStore({
  reducer: {
    auth: authReducer,
    projects: projectReducer,
    verifications: verificationReducer,
  },
});

export default store;
