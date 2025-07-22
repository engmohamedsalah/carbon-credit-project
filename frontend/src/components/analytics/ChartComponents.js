import React from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
} from 'chart.js';
import { Line, Bar, Doughnut, Pie } from 'react-chartjs-2';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement
);

// Common chart options
const commonOptions = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: {
      position: 'top',
    },
  },
};

export const ModelPerformanceChart = ({ data }) => {
  const chartData = {
    labels: data?.accuracy_trends?.map(d => d.date) || [],
    datasets: [
      {
        label: 'Forest Cover Model',
        data: data?.accuracy_trends?.map(d => (d.forest_cover * 100).toFixed(1)) || [],
        borderColor: 'rgb(76, 175, 80)',
        backgroundColor: 'rgba(76, 175, 80, 0.1)',
        tension: 0.4,
      },
      {
        label: 'Change Detection Model',
        data: data?.accuracy_trends?.map(d => (d.change_detection * 100).toFixed(1)) || [],
        borderColor: 'rgb(33, 150, 243)',
        backgroundColor: 'rgba(33, 150, 243, 0.1)',
        tension: 0.4,
      },
      {
        label: 'Ensemble Model',
        data: data?.accuracy_trends?.map(d => (d.ensemble * 100).toFixed(1)) || [],
        borderColor: 'rgb(156, 39, 176)',
        backgroundColor: 'rgba(156, 39, 176, 0.1)',
        tension: 0.4,
      },
    ],
  };

  const options = {
    ...commonOptions,
    scales: {
      y: {
        beginAtZero: false,
        min: 80,
        max: 100,
        ticks: {
          callback: function(value) {
            return value + '%';
          }
        }
      },
    },
    plugins: {
      ...commonOptions.plugins,
      title: {
        display: true,
        text: 'ML Model Accuracy Trends (Last 90 Days)',
      },
    },
  };

  return <Line data={chartData} options={options} />;
};

export const ProjectStatusChart = ({ data }) => {
  const statusData = data?.project_status || {};
  
  const chartData = {
    labels: Object.keys(statusData),
    datasets: [
      {
        data: Object.values(statusData),
        backgroundColor: [
          '#FF9800', // Pending - Orange
          '#4CAF50', // Verified - Green
          '#2196F3', // In Progress - Blue
          '#F44336', // Rejected - Red
        ],
        borderWidth: 2,
        borderColor: '#fff',
      },
    ],
  };

  const options = {
    ...commonOptions,
    plugins: {
      ...commonOptions.plugins,
      title: {
        display: true,
        text: 'Project Status Distribution',
      },
    },
  };

  return <Doughnut data={chartData} options={options} />;
};

export const CarbonImpactChart = ({ data }) => {
  const chartData = {
    labels: data?.impact_by_type?.map(d => d.type) || [],
    datasets: [
      {
        label: 'Carbon Impact (tonnes CO₂)',
        data: data?.impact_by_type?.map(d => d.impact) || [],
        backgroundColor: [
          'rgba(76, 175, 80, 0.8)',
          'rgba(33, 150, 243, 0.8)',
          'rgba(156, 39, 176, 0.8)',
          'rgba(255, 152, 0, 0.8)',
          'rgba(244, 67, 54, 0.8)',
        ],
        borderColor: [
          'rgb(76, 175, 80)',
          'rgb(33, 150, 243)',
          'rgb(156, 39, 176)',
          'rgb(255, 152, 0)',
          'rgb(244, 67, 54)',
        ],
        borderWidth: 1,
      },
    ],
  };

  const options = {
    ...commonOptions,
    plugins: {
      ...commonOptions.plugins,
      title: {
        display: true,
        text: 'Carbon Impact by Project Type',
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        ticks: {
          callback: function(value) {
            return value.toLocaleString() + ' tonnes';
          }
        }
      },
    },
  };

  return <Bar data={chartData} options={options} />;
};

export const MonthlyTrendsChart = ({ data }) => {
  const chartData = {
    labels: data?.monthly_trends?.map(d => d.month) || [],
    datasets: [
      {
        label: 'Carbon Impact (tonnes CO₂)',
        data: data?.monthly_trends?.map(d => d.impact) || [],
        borderColor: 'rgb(76, 175, 80)',
        backgroundColor: 'rgba(76, 175, 80, 0.1)',
        tension: 0.4,
        yAxisID: 'y',
      },
      {
        label: 'Verifications Count',
        data: data?.monthly_trends?.map(d => d.verifications) || [],
        borderColor: 'rgb(255, 152, 0)',
        backgroundColor: 'rgba(255, 152, 0, 0.1)',
        tension: 0.4,
        yAxisID: 'y1',
      },
    ],
  };

  const options = {
    ...commonOptions,
    interaction: {
      mode: 'index',
      intersect: false,
    },
    plugins: {
      ...commonOptions.plugins,
      title: {
        display: true,
        text: 'Monthly Carbon Impact & Verification Trends',
      },
    },
    scales: {
      x: {
        display: true,
        title: {
          display: true,
          text: 'Month'
        }
      },
      y: {
        type: 'linear',
        display: true,
        position: 'left',
        title: {
          display: true,
          text: 'Carbon Impact (tonnes)'
        }
      },
      y1: {
        type: 'linear',
        display: true,
        position: 'right',
        title: {
          display: true,
          text: 'Verifications'
        },
        grid: {
          drawOnChartArea: false,
        },
      },
    },
  };

  return <Line data={chartData} options={options} />;
};

export const ConfidenceDistributionChart = ({ data }) => {
  const distribution = data?.confidence_distribution || {};
  
  const chartData = {
    labels: ['High Confidence (>90%)', 'Medium Confidence (70-90%)', 'Low Confidence (<70%)'],
    datasets: [
      {
        data: [
          distribution.high_confidence || 0,
          distribution.medium_confidence || 0,
          distribution.low_confidence || 0
        ],
        backgroundColor: [
          'rgba(76, 175, 80, 0.8)', // Green for high
          'rgba(255, 152, 0, 0.8)',  // Orange for medium
          'rgba(244, 67, 54, 0.8)',  // Red for low
        ],
        borderColor: [
          'rgb(76, 175, 80)',
          'rgb(255, 152, 0)',
          'rgb(244, 67, 54)',
        ],
        borderWidth: 2,
      },
    ],
  };

  const options = {
    ...commonOptions,
    plugins: {
      ...commonOptions.plugins,
      title: {
        display: true,
        text: `AI Confidence Distribution (Avg: ${(distribution.average_confidence * 100 || 0).toFixed(1)}%)`,
      },
    },
  };

  return <Pie data={chartData} options={options} />;
};

export const DailyActivityChart = ({ data }) => {
  const activityData = data?.daily_activity || {};
  const dates = Object.keys(activityData).slice(-14); // Last 14 days
  const counts = dates.map(date => activityData[date]);

  const chartData = {
    labels: dates,
    datasets: [
      {
        label: 'New Projects',
        data: counts,
        backgroundColor: 'rgba(33, 150, 243, 0.8)',
        borderColor: 'rgb(33, 150, 243)',
        borderWidth: 1,
      },
    ],
  };

  const options = {
    ...commonOptions,
    plugins: {
      ...commonOptions.plugins,
      title: {
        display: true,
        text: 'Daily Project Creation Activity (Last 14 Days)',
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        ticks: {
          stepSize: 1,
        }
      },
    },
  };

  return <Bar data={chartData} options={options} />;
}; 