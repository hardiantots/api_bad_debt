module.exports = {
  apps: [
    {
      name: "api_bad_debt",
      cwd: __dirname,
      script: "./start_api.sh",
      interpreter: "bash",
      exec_mode: "fork",
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: "1G",
      kill_timeout: 10000,
      env: {
        PYTHONUNBUFFERED: "1",
        API_HOST: "0.0.0.0",
        API_PORT: "8000",
        UVICORN_RELOAD: "false",
      },
      error_file: "./local_data/logs/pm2-error.log",
      out_file: "./local_data/logs/pm2-out.log",
      merge_logs: true,
      time: true,
    },
  ],
};
