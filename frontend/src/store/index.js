import { configureStore } from '@reduxjs/toolkit';
import authReducer from './authSlice';
import projectReducer from './projectSlice';
import verificationReducer from './verificationSlice';
import xaiReducer from './xaiSlice';
import iotReducer from './iotSlice';

const store = configureStore({
  reducer: {
    auth: authReducer,
    projects: projectReducer,
    verifications: verificationReducer,
    xai: xaiReducer,
    iot: iotReducer,
  },
});

export default store;
