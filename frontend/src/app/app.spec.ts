import { TestBed } from '@angular/core/testing';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { provideHttpClient } from '@angular/common/http';
import { App } from './app';

describe('App', () => {
  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [App],
      providers: [provideHttpClient(), provideHttpClientTesting()],
    }).compileComponents();
  });

  afterEach(() => {
    TestBed.inject(HttpTestingController).verify();
  });

  it('should create the app', () => {
    const fixture = TestBed.createComponent(App);
    TestBed.inject(HttpTestingController)
      .expectOne('/api/health')
      .flush({ status: 'ok', service: 'api', timestamp: '2026-01-01T00:00:00Z' });
    const app = fixture.componentInstance;
    expect(app).toBeTruthy();
  });

  it('should render the foundation heading and online API state', async () => {
    const fixture = TestBed.createComponent(App);
    const request = TestBed.inject(HttpTestingController).expectOne('/api/health');
    request.flush({ status: 'ok', service: 'api', timestamp: '2026-01-01T00:00:00Z' });
    await fixture.whenStable();
    fixture.detectChanges();
    const compiled = fixture.nativeElement as HTMLElement;
    expect(compiled.querySelector('h1')?.textContent).toContain('Build grounded questions');
    expect(compiled.querySelector('.status')?.textContent).toContain('FastAPI foundation is responding.');
  });
});
