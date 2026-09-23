import { HttpClient } from '@angular/common/http';
import { Component, inject, signal } from '@angular/core';
import { firstValueFrom } from 'rxjs';

interface HealthResponse {
  status: 'ok';
  service: 'api';
  timestamp: string;
}

@Component({
  selector: 'app-root',
  imports: [],
  templateUrl: './app.html',
  styleUrl: './app.scss'
})
export class App {
  private readonly http = inject(HttpClient);
  protected readonly apiStatus = signal<'checking' | 'online' | 'offline'>('checking');
  protected readonly apiMessage = signal('Checking the API health endpoint...');

  constructor() {
    void this.checkApiHealth();
  }

  private async checkApiHealth(): Promise<void> {
    try {
      const health = await firstValueFrom(this.http.get<HealthResponse>('/api/health'));
      if (health.status !== 'ok' || health.service !== 'api') {
        throw new Error('Unexpected health response');
      }
      this.apiStatus.set('online');
      this.apiMessage.set('FastAPI foundation is responding.');
    } catch {
      this.apiStatus.set('offline');
      this.apiMessage.set('The API is not reachable. Start the backend to continue.');
    }
  }
}
