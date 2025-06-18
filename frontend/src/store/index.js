import { configureStore } from '@reduxjs/toolkit';
import authReducer from './authSlice';
import projectReducer from './projectSlice';
import verificationReducer from './verificationSlice';
import xaiReducer from './xaiSlice';

const store = configureStore({
  reducer: {
    auth: authReducer,
    projects: projectReducer,
    verifications: verificationReducer,
    xai: xaiReducer,
  },
});

export default store;
